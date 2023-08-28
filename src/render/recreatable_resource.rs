use std::{collections::VecDeque, borrow::Cow};

use crate::render;

pub type RecreatableBuffer = Recreatable<render::Buffer>;
pub type RecreatableImage = Recreatable<render::Image>;

struct OldResource<R> {
    resource: R,
    last_used_frame_index: Option<usize>,
}

pub struct Recreatable<R: render::RenderResource> {
    name: Cow<'static, str>,
    current_resource: R,
    current_desc: R::Desc,

    last_used_frame_index: Option<usize>,
    old: VecDeque<OldResource<R>>,
}

impl<R> Recreatable<R>
where 
    R: render::RenderResource,
    R::Desc: std::cmp::Eq,
{
    pub fn new(context: &render::Context, name: Cow<'static, str>, desc: R::Desc) -> Self {   
        let current = R::create(&context.device, &context.descriptors, name.clone(), &desc, None);
        Self {
            last_used_frame_index: None,
            name,
            current_resource: current,
            old: VecDeque::new(),
            current_desc: desc,
        }
    }

    pub fn recreate(&mut self, context: &render::Context) -> bool {
        puffin::profile_function!(&self.name);
        if self.current_desc == *self.current_resource.desc() {
            return false;
        }

        // slow but only happens on resizes so it's fine
        let mut tmp = R::create(&context.device, &context.descriptors, self.name.clone(), &self.current_desc, None);
        std::mem::swap(&mut tmp, &mut self.current_resource);
        
        self.old.push_back(OldResource {
            resource: tmp,
            last_used_frame_index: self.last_used_frame_index,
        });

        true
    }

    pub fn desc(&self) -> &R::Desc {
        &self.current_desc
    }

    pub fn desc_mut(&mut self) -> &mut R::Desc {
        &mut self.current_desc
    }

    pub fn get_current(&mut self,context: &mut render::Context) -> &R {
        puffin::profile_function!(&self.name);
        if let Some(old_resource) = self.old.front() {
            if old_resource.last_used_frame_index.is_none() ||
               old_resource.last_used_frame_index == Some(context.frame_index())
            {
                // slow but only happens after resizes so it's fine
                old_resource.resource.destroy(&context.device, &context.descriptors);
                self.old.pop_front();
            }
        }

        self.last_used_frame_index = Some(context.frame_index());
        &self.current_resource
    }

    pub fn destroy(&mut self, context: &render::Context) {
        puffin::profile_function!(&self.name);
        self.current_resource.destroy(&context.device, &context.descriptors);

        while let Some(old_resource) = self.old.pop_front() {
            old_resource.resource.destroy(&context.device, &context.descriptors);
        }
    }
}