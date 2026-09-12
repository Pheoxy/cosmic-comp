//! Renderer-generic compositor chrome.
//!
//! Callers construct [`CosmicChromeElement`] and draw it with [`RenderElement<R>`]. Glow/GLES
//! uses the rounded pixel shader when a glow frame exists. Everything else - Vulkan, Pixman, or
//! any other `Frame`-implementing renderer - gets a logical-space solid fill, a four-rect border,
//! a shadow via the renderer-generic `Frame::draw_shadow` (a real blurred shader on Vulkan, that
//! method's own stacked-fill approximation elsewhere), or a no-op.

use glam::{Affine2, Mat3, Vec2};
use smithay::{
    backend::renderer::{
        Color32F, Frame, Renderer, ShadowParameters,
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

/// Rounded GLES shader, solid fill, axis-aligned border, stacked shadow, or a no-op.
#[derive(Debug, Clone)]
pub enum CosmicChromeElement {
    Solid(SolidChromeElement),
    Border(BorderChromeElement),
    Shadow(GenericShadowElement),
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

    pub fn border(geo: Rectangle<i32, Local>, thickness: u8, alpha: f32, color: [f32; 3]) -> Self {
        Self::Border(BorderChromeElement::new(
            geo,
            thickness,
            Color32F::new(color[0], color[1], color[2], alpha),
        ))
    }

    pub fn shader(elem: PixelShaderElement) -> Self {
        Self::Shader(elem)
    }

    pub fn skip() -> Self {
        Self::Skip(SkipChromeElement::new())
    }

    /// Drop shadow for renderers without the rounded GLES shader, via [`Frame::draw_shadow`].
    ///
    /// Vulkan overrides that with a real blurred, rounded-corner shader; other renderers get the
    /// method's generic stacked-fill default. Either way this is the single non-GLES shadow path,
    /// so there is no separate CPU-side approximation to keep in sync with the shader here.
    #[allow(clippy::too_many_arguments)]
    pub fn shadow(
        geo: Rectangle<i32, Local>,
        input_to_geo: Mat3,
        window_input_to_geo: Mat3,
        color: [f32; 4],
        sigma: f32,
        geo_size: [f32; 2],
        corner_radius: [f32; 4],
        window_geo_size: [f32; 2],
        window_corner_radius: [f32; 4],
        alpha: f32,
    ) -> Self {
        Self::Shadow(GenericShadowElement::new(
            geo,
            input_to_geo,
            window_input_to_geo,
            color,
            sigma,
            geo_size,
            corner_radius,
            window_geo_size,
            window_corner_radius,
            alpha,
        ))
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

/// Axis-aligned border drawn with four `draw_solid` rects (Vulkan/Pixman focus rings).
#[derive(Debug, Clone)]
pub struct BorderChromeElement {
    id: Id,
    geo: Rectangle<i32, Logical>,
    thickness: u8,
    color: Color32F,
    commit: CommitCounter,
}

impl BorderChromeElement {
    pub fn new(geo: Rectangle<i32, Local>, thickness: u8, color: Color32F) -> Self {
        Self {
            id: Id::new(),
            geo: geo.as_logical(),
            thickness,
            color,
            commit: CommitCounter::default(),
        }
    }
}

/// Drop shadow drawn via the renderer-generic [`Frame::draw_shadow`] - a real blurred shader under
/// Vulkan, a stacked-fill approximation (that method's own default) under everything else.
///
/// `input_to_geo`/`window_input_to_geo` are the same `[0, 1]`-normalized-local-space-to-geo-space
/// matrices the GLES shader path (`ShadowShader::element`) builds; unlike that path, this element
/// has no vertex-interpolated local coordinate to feed them; `draw` composes them with a
/// `dst`-derived pixel-to-local mapping at draw time instead (see [`ShadowParameters`]'s docs on
/// `pixel_to_geo`).
#[derive(Debug, Clone)]
pub struct GenericShadowElement {
    id: Id,
    geo: Rectangle<i32, Logical>,
    input_to_geo: Mat3,
    window_input_to_geo: Mat3,
    color: [f32; 4],
    sigma: f32,
    geo_size: [f32; 2],
    corner_radius: [f32; 4],
    window_geo_size: [f32; 2],
    window_corner_radius: [f32; 4],
    alpha: f32,
    commit: CommitCounter,
}

impl GenericShadowElement {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        geo: Rectangle<i32, Local>,
        input_to_geo: Mat3,
        window_input_to_geo: Mat3,
        color: [f32; 4],
        sigma: f32,
        geo_size: [f32; 2],
        corner_radius: [f32; 4],
        window_geo_size: [f32; 2],
        window_corner_radius: [f32; 4],
        alpha: f32,
    ) -> Self {
        Self {
            id: Id::new(),
            geo: geo.as_logical(),
            input_to_geo,
            window_input_to_geo,
            color,
            sigma,
            geo_size,
            corner_radius,
            window_geo_size,
            window_corner_radius,
            alpha,
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

impl Element for BorderChromeElement {
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

    fn alpha(&self) -> f32 {
        self.color.a()
    }

    fn kind(&self) -> Kind {
        Kind::Unspecified
    }
}

impl Element for GenericShadowElement {
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

    fn alpha(&self) -> f32 {
        self.alpha
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
            Self::Border(elem) => elem.id(),
            Self::Shadow(elem) => elem.id(),
            Self::Shader(elem) => elem.id(),
            Self::Skip(elem) => elem.id(),
        }
    }

    fn current_commit(&self) -> CommitCounter {
        match self {
            Self::Solid(elem) => elem.current_commit(),
            Self::Border(elem) => elem.current_commit(),
            Self::Shadow(elem) => elem.current_commit(),
            Self::Shader(elem) => elem.current_commit(),
            Self::Skip(elem) => elem.current_commit(),
        }
    }

    fn src(&self) -> Rectangle<f64, BufferCoords> {
        match self {
            Self::Solid(elem) => elem.src(),
            Self::Border(elem) => elem.src(),
            Self::Shadow(elem) => elem.src(),
            Self::Shader(elem) => elem.src(),
            Self::Skip(elem) => elem.src(),
        }
    }

    fn geometry(&self, scale: Scale<f64>) -> Rectangle<i32, Physical> {
        match self {
            Self::Solid(elem) => elem.geometry(scale),
            Self::Border(elem) => elem.geometry(scale),
            Self::Shadow(elem) => elem.geometry(scale),
            Self::Shader(elem) => elem.geometry(scale),
            Self::Skip(elem) => elem.geometry(scale),
        }
    }

    fn location(&self, scale: Scale<f64>) -> Point<i32, Physical> {
        match self {
            Self::Solid(elem) => elem.location(scale),
            Self::Border(elem) => elem.location(scale),
            Self::Shadow(elem) => elem.location(scale),
            Self::Shader(elem) => elem.location(scale),
            Self::Skip(elem) => elem.location(scale),
        }
    }

    fn transform(&self) -> Transform {
        match self {
            Self::Solid(elem) => elem.transform(),
            Self::Border(elem) => elem.transform(),
            Self::Shadow(elem) => elem.transform(),
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
            Self::Border(elem) => elem.damage_since(scale, commit),
            Self::Shadow(elem) => elem.damage_since(scale, commit),
            Self::Shader(elem) => elem.damage_since(scale, commit),
            Self::Skip(elem) => elem.damage_since(scale, commit),
        }
    }

    fn opaque_regions(&self, scale: Scale<f64>) -> OpaqueRegions<i32, Physical> {
        match self {
            Self::Solid(elem) => elem.opaque_regions(scale),
            Self::Border(elem) => elem.opaque_regions(scale),
            Self::Shadow(elem) => elem.opaque_regions(scale),
            Self::Shader(elem) => elem.opaque_regions(scale),
            Self::Skip(elem) => elem.opaque_regions(scale),
        }
    }

    fn alpha(&self) -> f32 {
        match self {
            Self::Solid(elem) => elem.alpha(),
            Self::Border(elem) => elem.alpha(),
            Self::Shadow(elem) => elem.alpha(),
            Self::Shader(elem) => elem.alpha(),
            Self::Skip(_) => 0.0,
        }
    }

    fn kind(&self) -> Kind {
        match self {
            Self::Solid(elem) => elem.kind(),
            Self::Border(elem) => elem.kind(),
            Self::Shadow(elem) => elem.kind(),
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

impl<R: Renderer> RenderElement<R> for BorderChromeElement {
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
        let t = if self.geo.size.h > 0 {
            ((self.thickness as i32) * dst.size.h / self.geo.size.h).max(1)
        } else {
            1
        };
        let t = t.min(dst.size.w.max(1) / 2).min(dst.size.h.max(1) / 2);
        let sides = [
            Rectangle::new(dst.loc, (dst.size.w, t).into()),
            Rectangle::new(
                (dst.loc.x, dst.loc.y + dst.size.h - t).into(),
                (dst.size.w, t).into(),
            ),
            Rectangle::new(
                (dst.loc.x, dst.loc.y + t).into(),
                (t, dst.size.h - t * 2).into(),
            ),
            Rectangle::new(
                (dst.loc.x + dst.size.w - t, dst.loc.y + t).into(),
                (t, dst.size.h - t * 2).into(),
            ),
        ];
        for side in sides {
            if side.size.w <= 0 || side.size.h <= 0 {
                continue;
            }
            frame.draw_solid(side, damage, self.color)?;
        }
        Ok(())
    }
}

impl<R: Renderer> RenderElement<R> for GenericShadowElement {
    fn draw(
        &self,
        frame: &mut R::Frame<'_, '_>,
        _src: Rectangle<f64, BufferCoords>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        _opaque_regions: &[Rectangle<i32, Physical>],
        _cache: Option<&UserDataMap>,
    ) -> Result<(), R::Error> {
        if self.alpha <= 0.0 || dst.size.w <= 0 || dst.size.h <= 0 || damage.is_empty() {
            return Ok(());
        }
        // `input_to_geo`/`window_input_to_geo` map a `[0, 1]`-normalized local coordinate (as a
        // GLES vertex shader would interpolate across this element's own quad) into geo space.
        // This element has no such vertex-interpolated coordinate, so compose that with a
        // `dst`-derived mapping from absolute framebuffer pixels to the same `[0, 1]` local space,
        // giving `Frame::draw_shadow` implementations a direct pixel-to-geo matrix instead.
        let pixel_to_local = Mat3::from(
            Affine2::from_scale(Vec2::new(1.0 / dst.size.w as f32, 1.0 / dst.size.h as f32))
                * Affine2::from_translation(Vec2::new(-dst.loc.x as f32, -dst.loc.y as f32)),
        );
        let pixel_to_geo = self.input_to_geo * pixel_to_local;
        let pixel_to_window_geo = self.window_input_to_geo * pixel_to_local;
        frame.draw_shadow(
            dst,
            damage,
            ShadowParameters {
                pixel_to_geo: pixel_to_geo.to_cols_array(),
                pixel_to_window_geo: pixel_to_window_geo.to_cols_array(),
                color: self.color,
                sigma: self.sigma,
                geo_size: self.geo_size,
                corner_radius: self.corner_radius,
                window_geo_size: self.window_geo_size,
                window_corner_radius: self.window_corner_radius,
                alpha: self.alpha,
            },
        )
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
            Self::Border(elem) => {
                RenderElement::<R>::draw(elem, frame, src, dst, damage, opaque_regions, cache)
            }
            Self::Shadow(elem) => {
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
            Self::Border(elem) => elem.underlying_storage(renderer),
            Self::Shadow(elem) => elem.underlying_storage(renderer),
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
