//! Renderer-generic compositor chrome.
//!
//! Callers construct [`CosmicChromeElement`] and draw it with [`RenderElement<R>`]. Glow/GLES
//! uses the rounded pixel shader when a glow frame exists. Vulkan, Pixman, and any other
//! `Frame::draw_solid` renderer get a logical-space solid fill, a four-rect border,
//! stacked offset fills for shadows, or a no-op. Rounded corners, blur, and gaussian
//! shadows wait on Vulkan pipelines.

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

/// Rounded GLES shader, solid fill, axis-aligned border, stacked shadow, or a no-op.
#[derive(Debug, Clone)]
pub enum CosmicChromeElement {
    Solid(SolidChromeElement),
    Border(BorderChromeElement),
    Shadow(ShadowFillElement),
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

    /// Drop-shadow approximation for renderers without the rounded GLES shader.
    ///
    /// Matches the GLES shader's offset/spread/softness extents with stacked fills.
    /// The window is drawn on top and covers the center.
    pub fn stacked_shadow(geo: Rectangle<i32, Local>, alpha: f32, dark_mode: bool) -> Self {
        Self::Shadow(ShadowFillElement::new(geo, alpha, dark_mode))
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

/// Offset stacked fills approximating the GLES gaussian drop shadow.
#[derive(Debug, Clone)]
pub struct ShadowFillElement {
    id: Id,
    geo: Rectangle<i32, Logical>,
    layers: Vec<(Rectangle<i32, Logical>, f32)>,
    commit: CommitCounter,
}

impl ShadowFillElement {
    pub fn new(window: Rectangle<i32, Local>, alpha: f32, dark_mode: bool) -> Self {
        let window = window.as_logical();
        let base = alpha * if dark_mode { 0.45 } else { 0.35 };
        // Shader: offset [0, 5], spread 5, softness 25 (sigma 12.5, width ~38).
        let offset_y = 5;
        let layers_def = [(40, 0.15), (24, 0.28), (12, 0.50), (6, 0.80)];
        let mut layers = Vec::with_capacity(layers_def.len());
        let mut outer = window;
        for (expand, weight) in layers_def {
            let geo = Rectangle::new(
                (window.loc.x - expand, window.loc.y - expand + offset_y).into(),
                (
                    window.size.w.saturating_add(expand * 2),
                    window.size.h.saturating_add(expand * 2),
                )
                    .into(),
            );
            if geo.size.w >= outer.size.w && geo.size.h >= outer.size.h {
                outer = geo;
            }
            layers.push((geo, (base * weight).clamp(0.0, 1.0)));
        }
        Self {
            id: Id::new(),
            geo: outer,
            layers,
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

impl Element for ShadowFillElement {
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
        self.layers.last().map(|(_, a)| *a).unwrap_or(0.0)
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

impl<R: Renderer> RenderElement<R> for ShadowFillElement {
    fn draw(
        &self,
        frame: &mut R::Frame<'_, '_>,
        _src: Rectangle<f64, BufferCoords>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        _opaque_regions: &[Rectangle<i32, Physical>],
        _cache: Option<&UserDataMap>,
    ) -> Result<(), R::Error> {
        if dst.size.w <= 0 || dst.size.h <= 0 || damage.is_empty() {
            return Ok(());
        }
        let sx = dst.size.w as f64 / self.geo.size.w.max(1) as f64;
        let sy = dst.size.h as f64 / self.geo.size.h.max(1) as f64;
        for (layer, alpha) in &self.layers {
            if *alpha <= 0.0 {
                continue;
            }
            let loc = (
                dst.loc.x + ((layer.loc.x - self.geo.loc.x) as f64 * sx).round() as i32,
                dst.loc.y + ((layer.loc.y - self.geo.loc.y) as f64 * sy).round() as i32,
            );
            let size = (
                (layer.size.w as f64 * sx).round() as i32,
                (layer.size.h as f64 * sy).round() as i32,
            );
            let rect = Rectangle::new(loc.into(), size.into());
            if rect.size.w <= 0 || rect.size.h <= 0 {
                continue;
            }
            frame.draw_solid(rect, damage, Color32F::new(0.0, 0.0, 0.0, *alpha))?;
        }
        Ok(())
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
