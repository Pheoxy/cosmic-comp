//! Renderer-generic compositor chrome.
//!
//! Callers construct [`CosmicChromeElement`] and draw it with [`RenderElement<R>`]. Glow/GLES
//! uses the rounded pixel shader when a glow frame exists. Vulkan, Pixman, and any other
//! `Frame::draw_solid` renderer get a logical-space solid fill (fills) or a no-op (outlines
//! and shadows until those pipelines exist).

use smithay::{
    backend::renderer::{
        Color32F, Frame, Renderer,
        element::{Element, Id, Kind, RenderElement, UnderlyingStorage},
        gles::element::PixelShaderElement,
        glow::GlowRenderer,
        utils::{CommitCounter, DamageSet, OpaqueRegions},
    },
    utils::{
        Buffer as BufferCoords, Logical, Physical, Point, Rectangle, Scale, Transform,
        user_data::UserDataMap,
    },
};

use crate::utils::prelude::{Local, RectLocalExt};

use super::element::AsGlowRenderer;

/// Rounded GLES shader, solid fill, or a no-op.
#[derive(Debug, Clone)]
pub enum CosmicChromeElement {
    Solid(SolidChromeElement),
    Shader(PixelShaderElement),
    Skip(SkipChromeElement),
}

impl CosmicChromeElement {
    pub fn fill(geo: Rectangle<i32, Local>, alpha: f32, color: [f32; 3]) -> Self {
        Self::Solid(SolidChromeElement::new(
            geo,
            Color32F::new(color[0], color[1], color[2], alpha),
        ))
    }

    pub fn shader(elem: PixelShaderElement) -> Self {
        Self::Shader(elem)
    }

    pub fn skip() -> Self {
        Self::Skip(SkipChromeElement::new())
    }
}

#[derive(Debug, Clone)]
pub struct SolidChromeElement {
    id: Id,
    geo: Rectangle<i32, Logical>,
    color: Color32F,
    commit: CommitCounter,
}

impl SolidChromeElement {
    pub fn new(geo: Rectangle<i32, Local>, color: Color32F) -> Self {
        Self {
            id: Id::new(),
            geo: geo.as_logical(),
            color,
            commit: CommitCounter::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SkipChromeElement {
    id: Id,
}

impl SkipChromeElement {
    pub fn new() -> Self {
        Self { id: Id::new() }
    }
}

impl Element for SolidChromeElement {
    fn id(&self) -> &Id {
        &self.id
    }

    fn current_commit(&self) -> CommitCounter {
        self.commit
    }

    fn src(&self) -> Rectangle<f64, BufferCoords> {
        Rectangle::from_size((1.0, 1.0).into())
    }

    fn geometry(&self, scale: Scale<f64>) -> Rectangle<i32, Physical> {
        self.geo.to_physical_precise_round(scale)
    }

    fn opaque_regions(&self, scale: Scale<f64>) -> OpaqueRegions<i32, Physical> {
        if self.color.is_opaque() {
            OpaqueRegions::from_slice(&[Rectangle::from_size(self.geometry(scale).size)])
        } else {
            OpaqueRegions::default()
        }
    }

    fn alpha(&self) -> f32 {
        self.color.a()
    }

    fn kind(&self) -> Kind {
        Kind::Unspecified
    }
}

impl Element for SkipChromeElement {
    fn id(&self) -> &Id {
        &self.id
    }

    fn current_commit(&self) -> CommitCounter {
        CommitCounter::default()
    }

    fn src(&self) -> Rectangle<f64, BufferCoords> {
        Rectangle::from_size((1.0, 1.0).into())
    }

    fn geometry(&self, _scale: Scale<f64>) -> Rectangle<i32, Physical> {
        Rectangle::from_size((0, 0).into())
    }

    fn kind(&self) -> Kind {
        Kind::Unspecified
    }
}

impl Element for CosmicChromeElement {
    fn id(&self) -> &Id {
        match self {
            Self::Solid(elem) => elem.id(),
            Self::Shader(elem) => elem.id(),
            Self::Skip(elem) => elem.id(),
        }
    }

    fn current_commit(&self) -> CommitCounter {
        match self {
            Self::Solid(elem) => elem.current_commit(),
            Self::Shader(elem) => elem.current_commit(),
            Self::Skip(elem) => elem.current_commit(),
        }
    }

    fn src(&self) -> Rectangle<f64, BufferCoords> {
        match self {
            Self::Solid(elem) => elem.src(),
            Self::Shader(elem) => elem.src(),
            Self::Skip(elem) => elem.src(),
        }
    }

    fn geometry(&self, scale: Scale<f64>) -> Rectangle<i32, Physical> {
        match self {
            Self::Solid(elem) => elem.geometry(scale),
            Self::Shader(elem) => elem.geometry(scale),
            Self::Skip(elem) => elem.geometry(scale),
        }
    }

    fn location(&self, scale: Scale<f64>) -> Point<i32, Physical> {
        match self {
            Self::Solid(elem) => elem.location(scale),
            Self::Shader(elem) => elem.location(scale),
            Self::Skip(elem) => elem.location(scale),
        }
    }

    fn transform(&self) -> Transform {
        match self {
            Self::Solid(elem) => elem.transform(),
            Self::Shader(elem) => elem.transform(),
            Self::Skip(elem) => elem.transform(),
        }
    }

    fn damage_since(
        &self,
        scale: Scale<f64>,
        commit: Option<CommitCounter>,
    ) -> DamageSet<i32, Physical> {
        match self {
            Self::Solid(elem) => elem.damage_since(scale, commit),
            Self::Shader(elem) => elem.damage_since(scale, commit),
            Self::Skip(elem) => elem.damage_since(scale, commit),
        }
    }

    fn opaque_regions(&self, scale: Scale<f64>) -> OpaqueRegions<i32, Physical> {
        match self {
            Self::Solid(elem) => elem.opaque_regions(scale),
            Self::Shader(elem) => elem.opaque_regions(scale),
            Self::Skip(elem) => elem.opaque_regions(scale),
        }
    }

    fn alpha(&self) -> f32 {
        match self {
            Self::Solid(elem) => elem.alpha(),
            Self::Shader(elem) => elem.alpha(),
            Self::Skip(_) => 0.0,
        }
    }

    fn kind(&self) -> Kind {
        match self {
            Self::Solid(elem) => elem.kind(),
            Self::Shader(elem) => elem.kind(),
            Self::Skip(elem) => elem.kind(),
        }
    }
}

impl<R: Renderer> RenderElement<R> for SolidChromeElement {
    fn draw(
        &self,
        frame: &mut R::Frame<'_, '_>,
        _src: Rectangle<f64, BufferCoords>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        _opaque_regions: &[Rectangle<i32, Physical>],
        _cache: Option<&UserDataMap>,
    ) -> Result<(), R::Error> {
        if self.color.a() <= 0.0 || dst.size.w <= 0 || dst.size.h <= 0 || damage.is_empty() {
            return Ok(());
        }
        frame.draw_solid(dst, damage, self.color)
    }
}

impl<R: Renderer> RenderElement<R> for SkipChromeElement {
    fn draw(
        &self,
        _frame: &mut R::Frame<'_, '_>,
        _src: Rectangle<f64, BufferCoords>,
        _dst: Rectangle<i32, Physical>,
        _damage: &[Rectangle<i32, Physical>],
        _opaque_regions: &[Rectangle<i32, Physical>],
        _cache: Option<&UserDataMap>,
    ) -> Result<(), R::Error> {
        Ok(())
    }
}

impl<R> RenderElement<R> for CosmicChromeElement
where
    R: AsGlowRenderer,
{
    fn draw(
        &self,
        frame: &mut R::Frame<'_, '_>,
        src: Rectangle<f64, BufferCoords>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        cache: Option<&UserDataMap>,
    ) -> Result<(), R::Error> {
        match self {
            Self::Solid(elem) => {
                RenderElement::<R>::draw(elem, frame, src, dst, damage, opaque_regions, cache)
            }
            Self::Shader(elem) => {
                let Some(glow_frame) = R::glow_frame_mut(frame) else {
                    return Ok(());
                };
                RenderElement::<GlowRenderer>::draw(
                    elem,
                    glow_frame,
                    src,
                    dst,
                    damage,
                    opaque_regions,
                    cache,
                )
                .map_err(R::from_gles_error)
            }
            Self::Skip(elem) => {
                RenderElement::<R>::draw(elem, frame, src, dst, damage, opaque_regions, cache)
            }
        }
    }

    fn underlying_storage(&self, renderer: &mut R) -> Option<UnderlyingStorage<'_>> {
        match self {
            Self::Solid(elem) => elem.underlying_storage(renderer),
            Self::Shader(elem) => {
                let glow = renderer.glow_renderer_mut()?;
                elem.underlying_storage(glow)
            }
            Self::Skip(elem) => elem.underlying_storage(renderer),
        }
    }
}

impl From<PixelShaderElement> for CosmicChromeElement {
    fn from(elem: PixelShaderElement) -> Self {
        Self::Shader(elem)
    }
}
