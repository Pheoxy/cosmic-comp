#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::len_without_is_empty,
    clippy::collapsible_match
)]
// SPDX-License-Identifier: GPL-3.0-only

use calloop::timer::{TimeoutAction, Timer};
use smithay::{
    reexports::{
        calloop::{EventLoop, Interest, Mode, PostAction, generic::Generic},
        wayland_server::{Display, DisplayHandle},
    },
    wayland::socket::ListeningSocketSource,
};

use anyhow::{Context, Result};
use state::{BackendData, LastRefresh, State};
use std::{
    env,
    ffi::OsString,
    os::unix::process::CommandExt,
    process,
    sync::Arc,
    time::{Duration, Instant},
};
use tracing::{error, info, warn};
use wayland::protocols::{
    keyboard_layout::KeyboardLayoutState, overlap_notify::OverlapNotifyState,
};

use crate::wayland::handlers::compositor::client_compositor_state;

use clap_lex::RawArgs;

use std::error::Error;

/// Set by `cosmic-greeter-start.sh` on the greeter's `cosmic-comp` instance only - gates the
/// handoff behavior below so a normal desktop session never takes this path on exit.
const HANDOFF_ENV_VAR: &str = "COSMIC_GREETER_HANDOFF";
/// Records the pid of a still-lingering handoff holder (see `start_handoff_holder`) so the
/// *next* greeter instance to start on the same VT can reap it - see `reap_stale_handoff_holder`.
const HANDOFF_PIDFILE: &str = "/run/cosmic-greeter/handoff.pid";

fn is_handoff_mode() -> bool {
    env::var_os(HANDOFF_ENV_VAR).is_some()
}

pub mod backend;
pub mod config;
pub mod dbus;
#[cfg(feature = "debug")]
pub mod debug;
pub mod hooks;
pub mod input;
pub mod libei;
mod logger;
pub mod session;
pub mod shell;
pub mod state;
#[cfg(feature = "systemd")]
pub mod systemd;
pub mod theme;
pub mod utils;
pub mod wayland;
pub mod xwayland;

#[cfg(feature = "profile-with-tracy")]
#[global_allocator]
static GLOBAL: profiling::tracy_client::ProfiledAllocator<std::alloc::System> =
    profiling::tracy_client::ProfiledAllocator::new(std::alloc::System, 10);

// called by the Xwayland source, either after starting or failing
impl State {
    fn notify_ready(&mut self) {
        // TODO: Don't notify again, but potentially import updated env-variables
        // into systemd and the session?
        self.ready.call_once(|| {
            // potentially tell systemd we are setup now
            if let state::BackendData::Kms(_) = &self.backend {
                #[cfg(feature = "systemd")]
                systemd::ready(&self.common);
                if let Err(err) = dbus::ready(&self.common) {
                    error!(?err, "Failed to update the D-Bus activation environment");
                }
            }

            // potentially tell the session we are setup now
            if let Err(err) =
                session::run_socket(self.common.event_loop_handle.clone(), &self.common)
            {
                warn!(?err, "Failed to setup cosmic-session communication");
            }

            self.common.kiosk_child = if let Some(mut command) = self.kiosk_command.take() {
                // Run command in kiosk mode
                command.envs(
                    session::get_env(&self.common).expect("WAYLAND_DISPLAY should be valid UTF-8"),
                );
                unsafe {
                    command.pre_exec(|| {
                        utils::rlimit::restore_nofile_limit();
                        Ok(())
                    })
                };

                info!("Running {:?}", command.get_program());
                command
                    .spawn()
                    .map_err(|err| {
                        // TODO: replace with `inspect_err` once stable
                        error!(?err, "Error running kiosk child.");
                        err
                    })
                    .ok()
            } else {
                None
            };
        });
    }
}

pub fn run(hooks: crate::hooks::Hooks) -> Result<(), Box<dyn Error>> {
    let raw_args = RawArgs::from_args();
    let mut cursor = raw_args.cursor();
    raw_args.next_os(&mut cursor);
    let git_hash = option_env!("GIT_HASH").unwrap_or("unknown");

    let mut kiosk_command = None;
    let mut with_xwayland = true;
    // Parse the arguments
    while let Some(arg) = raw_args.next_os(&mut cursor) {
        match arg.to_str() {
            Some("--help") | Some("-h") => {
                print_help(env!("CARGO_PKG_VERSION"), git_hash);
                return Ok(());
            }
            Some("--no-xwayland") => {
                tracing::info!("Running without Xwayland");
                with_xwayland = false;
            }
            Some("--version") | Some("-V") => {
                println!(
                    "cosmic-comp {} (git commit {})",
                    env!("CARGO_PKG_VERSION"),
                    git_hash
                );
                return Ok(());
            }
            _ => {
                let mut cmd = process::Command::new(arg);
                cmd.args(raw_args.remaining(&mut cursor));
                kiosk_command = Some(cmd);
            }
        }
    }

    // setup logger
    logger::init_logger()?;
    info!("Cosmic starting up!");

    if is_handoff_mode() {
        reap_stale_handoff_holder();
    }

    profiling::register_thread!("Main Thread");
    #[cfg(feature = "profile-with-tracy")]
    tracy_client::Client::start();

    utils::rlimit::increase_nofile_limit();
    // This needs to be done before any potential program launches
    // (e.g. Xwayland) as it handles passed file descriptors.
    if let Err(err) = session::setup_socket() {
        warn!("Session error: {:?}", err);
    };

    // init hook globals
    hooks::HOOKS.set(hooks)
        .expect("Hooks global has already been initialized. Running multiple instances of COSMIC in one process is not supported.");

    // init event loop
    let mut event_loop = EventLoop::try_new().with_context(|| "Failed to initialize event loop")?;
    // init wayland
    let (display, socket) = init_wayland_display(&mut event_loop)?;
    // init state
    let mut state = state::State::new(
        &display,
        socket,
        event_loop.handle(),
        event_loop.get_signal(),
        with_xwayland,
        kiosk_command,
    );
    // Set up the libei sender side before the backend spawns Xwayland.
    let ei_sender = libei::setup_ei(&event_loop.handle());
    state.common.dbus_state.set_ei_sender(ei_sender);

    // init backend
    backend::init_backend_auto(&display, &mut event_loop, &mut state)?;

    if let Err(err) = theme::watch_theme(event_loop.handle()) {
        warn!(?err, "Failed to watch theme");
    }

    // run the event loop
    event_loop.run(None, &mut state, |state| {
        // shall we shut down?
        if state.common.should_stop {
            info!("Shutting down");
            state.common.event_loop_signal.stop();
            state.common.event_loop_signal.wakeup();
            return;
        }

        // trigger routines
        let clients = state.common.shell.write().update_animations();
        {
            let dh = state.common.display_handle.clone();
            for client in clients.values() {
                client_compositor_state(client).blocker_cleared(state, &dh);
            }
        }

        refresh(state);

        {
            let shell = state.common.shell.read();
            if shell.animations_going() {
                for output in shell.outputs().cloned().collect::<Vec<_>>().into_iter() {
                    state.backend.schedule_render(&output);
                }
            }
        }

        // send out events
        let _ = state.common.display_handle.flush_clients();

        // check if kiosk child is running
        if let Some(child) = state.common.kiosk_child.as_mut() {
            match child.try_wait() {
                // Kiosk child exited with status
                Ok(Some(exit_status)) => {
                    info!("Command exited with status {:?}", exit_status);
                    // Stop cleanly so surface threads are joined before exit() (signal -> 1).
                    state.common.kiosk_exit_code = Some(exit_status.code().unwrap_or(1));
                    state.common.should_stop = true;
                }
                // Command still running
                Ok(None) => {}
                // Kiosk child disappeared, exiting with error
                Err(err) => {
                    warn!(?err, "Failed to wait for command");
                    state.common.kiosk_exit_code = Some(1);
                    state.common.should_stop = true;
                }
            }
        }
    })?;

    // kill kiosk child if loop exited
    if let Some(mut child) = state.common.kiosk_child.take() {
        let _ = child.kill();
    }

    let kiosk_exit_code = state.common.kiosk_exit_code;

    // Greeter handoff: on a clean login (kiosk child exited 0) hand the display off to the
    // incoming session instead of tearing it down here. greetd won't start the user session
    // until this process exits either way (it's not a subreaper and blocks in waitpid() on
    // exactly this pid), and systemd-logind unconditionally forces the VT back to the text
    // console the moment our logind/libseat control connection closes - with no check for
    // whether a successor has since taken the VT over - so the only way to avoid that blank
    // is to keep that connection (and our DRM fds) open past this point. See
    // `start_handoff_holder` for how.
    // `start_handoff_holder` only returns at all if `fork()` itself failed - both of its
    // success paths end in either an infinite sleep (the holder child) or `process::exit`
    // (this process, once the holder is confirmed running), so falling through to the normal
    // teardown below is exactly the right behavior on its one failure path too.
    if is_handoff_mode() && kiosk_exit_code == Some(0) {
        start_handoff_holder();
    }

    // Join surface threads before exit() so no thread is mid-eglCreateSync when
    // Mesa's atexit handlers run and corrupt the heap (issue #2375). Safe here
    // because the event loop has stopped; an unconditional join in Surface::Drop
    // would instead deadlock against apply_config_for_outputs.
    if let BackendData::Kms(kms) = &mut state.backend {
        // Release master first so the surface drop path skips its blocking commit.
        for device in kms.drm_devices.values_mut() {
            device.drm.pause();
        }
        for device in kms.drm_devices.values_mut() {
            for (_, surface) in device.inner.surfaces.drain() {
                surface.drop_and_join();
            }
        }
    }

    // drop eventloop & state before logger
    std::mem::drop(event_loop);
    std::mem::drop(state);

    if let Some(code) = kiosk_exit_code {
        process::exit(code);
    }

    Ok(())
}

/// Kills any handoff holder process left behind by a previous login on this VT (see
/// `start_handoff_holder`) and clears its pidfile. Only ever relevant right as a *new* greeter
/// instance starts, which only happens after a logout - i.e. at a point where the VT is already
/// resetting to a fresh greeter, so the kill's own `session_restore_vt` side effect (see
/// `start_handoff_holder`'s doc comment) isn't separately visible.
fn reap_stale_handoff_holder() {
    let Ok(contents) = std::fs::read_to_string(HANDOFF_PIDFILE) else {
        return;
    };
    if let Ok(pid) = contents.trim().parse::<i32>() {
        // The holder is a forked (never exec'd) child of a past `cosmic-comp`, so its /proc
        // comm is still "cosmic-comp" - check that before SIGKILLing a pid read from a file,
        // in case the pidfile is stale enough that the kernel has since reused that pid for an
        // unrelated process.
        let is_our_holder = std::fs::read_to_string(format!("/proc/{pid}/comm"))
            .is_ok_and(|comm| comm.trim() == "cosmic-comp");
        if is_our_holder {
            // SIGKILL: the holder does nothing but block in pause(), no cleanup needed or
            // wanted - it holding stale fds open is the whole point, right up until now.
            unsafe {
                libc::kill(pid, libc::SIGKILL);
            }
        }
    }
    let _ = std::fs::remove_file(HANDOFF_PIDFILE);
}

/// Forks a holder process that inherits every fd this process currently has open - most
/// importantly the DRM device fds and the logind/libseat session-control connection - and then
/// does nothing but block forever, while this process exits immediately without running any of
/// its usual teardown.
///
/// This matters because `systemd-logind`'s `session_restore_vt()` (which forces the VT back to
/// the text console) only ever fires when our logind session's *controller* connection actually
/// closes, or when the session record itself gets garbage-collected - never on a plain PAM
/// session stop, and never while some process (any process) keeps that connection open. As long
/// as the holder keeps breathing, our CRTC/connector/framebuffer state stays exactly as the
/// kernel and logind already see it, so the incoming session's own compositor reads it as
/// unchanged and its first frame is a plain page-flip instead of a disruptive blank-and-reset.
///
/// Deliberately does *not* wait for the incoming session to actually take over before exiting:
/// greetd only starts that session once this process (specifically, the one it originally
/// spawned) has exited, so there is nothing to wait for here - by construction, the incoming
/// session cannot even begin starting until after we've already handed off.
///
/// Returns only if `fork()` itself failed, so the caller can fall back to normal teardown; both
/// success paths (the holder child, and this process after recording the holder's pid) never
/// return.
fn start_handoff_holder() {
    match unsafe { libc::fork() } {
        -1 => {
            warn!("Handoff fork failed, falling back to normal shutdown");
        }
        0 => {
            // Child, immediately post-fork in a process that was multithreaded a moment ago:
            // fork() only duplicates the calling thread, so every other thread (and any lock
            // one of them happened to hold, heap allocator included) simply doesn't exist here
            // anymore - only async-signal-safe operations are valid. No allocation, no Rust
            // destructors, nothing but blocking forever holding the inherited fds open.
            loop {
                unsafe {
                    libc::pause();
                }
            }
        }
        pid => {
            let _ = std::fs::create_dir_all("/run/cosmic-greeter");
            if let Err(err) = std::fs::write(HANDOFF_PIDFILE, pid.to_string()) {
                warn!(?err, "Failed to record handoff holder pid");
            }
            info!(holder_pid = pid, "Handing display off to incoming session");
            // `std::process::exit` only skips Rust destructors - it still runs libc atexit
            // handlers, including Mesa's, with our surface threads still alive (the exact
            // heap-corruption scenario issue #2375's join-before-exit above exists to avoid,
            // except here it can also just hang: an atexit handler blocking on a lock a live
            // render thread holds means this process - and therefore greetd's waitpid on it,
            // and therefore the login - never completes). `_exit` bypasses atexit entirely,
            // which is what "exit without running any teardown" actually requires here.
            unsafe {
                libc::_exit(0);
            }
        }
    }
}

fn print_help(version: &str, git_rev: &str) {
    println!(
        r#"cosmic-comp {version} (git commit {git_rev})
System76 <info@system76.com>

Designed for the COSMIC™ desktop environment, cosmic-comp is a Wayland Compositor.

Project home page: https://github.com/pop-os/cosmic-comp

Options:
  -h, --help          Show this message
  --no-xwayland       Run without Xwayland
  -v, --version       Show the version of cosmic-comp"#
    );
}

fn init_wayland_display(
    event_loop: &mut EventLoop<state::State>,
) -> Result<(DisplayHandle, OsString)> {
    let display = Display::new().unwrap();
    let handle = display.handle();

    let source = ListeningSocketSource::new_auto().unwrap();
    let socket_name = source.socket_name().to_os_string();
    info!("Listening on {:?}", socket_name);

    event_loop
        .handle()
        .insert_source(source, |client_stream, _, state| {
            let client_state = state.new_client_state();
            if let Err(err) = state
                .common
                .display_handle
                .insert_client(client_stream, Arc::new(client_state))
            {
                warn!(?err, "Error adding wayland client")
            };
        })
        .with_context(|| "Failed to init the wayland socket source.")?;
    event_loop
        .handle()
        .insert_source(
            Generic::new(display, Interest::READ, Mode::Level),
            move |_, display, state| {
                // SAFETY: We don't drop the display
                match unsafe { display.get_mut().dispatch_clients(state) } {
                    Ok(_) => Ok(PostAction::Continue),
                    Err(err) => {
                        error!(?err, "I/O error on the Wayland display");
                        state.common.should_stop = true;
                        Err(err)
                    }
                }
            },
        )
        .with_context(|| "Failed to init the wayland event source.")?;

    Ok((handle, socket_name))
}

fn refresh(state: &mut State) {
    if matches!(state.last_refresh, LastRefresh::Scheduled(_)) {
        return;
    }

    if matches!(state.last_refresh, LastRefresh::At(instant) if Instant::now().duration_since(instant) < Duration::from_millis(150))
    {
        if let Ok(token) = state.common.event_loop_handle.insert_source(
            Timer::from_duration(Duration::from_millis(150)),
            |_, _, state| {
                state.last_refresh = LastRefresh::None;
                TimeoutAction::Drop
            },
        ) {
            state.last_refresh = LastRefresh::Scheduled(token);
            return;
        } else {
            warn!("Failed to schedule refresh");
        }
    }

    state.common.refresh();
    state::Common::refresh_focus(state);
    OverlapNotifyState::refresh(state);
    state.common.update_x11_stacking_order();
    KeyboardLayoutState::refresh(state);
    state.last_refresh = LastRefresh::At(Instant::now());
}
