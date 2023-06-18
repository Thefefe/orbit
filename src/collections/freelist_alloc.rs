use crate::collections::arena;

#[derive(Debug, Clone, Copy)]
pub struct BlockRange {
    pub start: usize,
    pub end: usize,
}

impl BlockRange {
    pub fn size(&self) -> usize {
        self.end - self.start
    }
}

struct Block {
    free: bool,
    range: BlockRange,
    prev_index: Option<arena::Index>,
    next_index: Option<arena::Index>,
}

#[derive(Clone, Copy)]
struct FreeBlock {
    index: arena::Index,
    size: usize,
}

pub struct FreeListAllocator {
    blocks: arena::Arena<Block>,
}

impl FreeListAllocator {
    pub fn new(size: usize) -> Self {
        let mut blocks = arena::Arena::new();

        blocks.insert(Block {
            free: true,
            range: BlockRange { start: 0, end: size },
            prev_index: None,
            next_index: None,
        });

        Self { blocks }
    }

    pub fn allocate(&mut self, size: usize) -> Option<(arena::Index, BlockRange)> {
        let (free_block_index, free_block) = self
            .blocks
            .iter()
            .filter(|(_, block)| block.free && block.range.end >= size)
            .min_by_key(|(_, block)| block.range.end)?;

        if free_block.range.size() == size {
            self.blocks[free_block_index].free = false;
            return Some((free_block_index, self.blocks[free_block_index].range));
        }

        let start = self.blocks[free_block_index].range.start;
        let prev_index = self.blocks[free_block_index].prev_index;

        let range = BlockRange { start, end: start + size };

        let new_block = self.blocks.insert(Block {
            free: false,
            range,
            prev_index,
            next_index: Some(free_block_index),
        });

        let free_block = &mut self.blocks[free_block_index];
        free_block.range.start += size;
        free_block.prev_index = Some(new_block);

        Some((new_block, range))
    }

    pub fn deallocate(&mut self, index: arena::Index) {
        if !self.blocks.has_index(index) {
            return;
        }

        let prev_free_index = self.blocks[index]
            .prev_index
            .filter(|&index| self.blocks[index].free);
        let next_free_index = self.blocks[index]
            .next_index
            .filter(|&index| self.blocks[index].free);

        self.blocks[index].free = true;

        if let Some(prev_free_index) = prev_free_index {
            let prev_block = self.blocks.remove(prev_free_index).unwrap();
            if let Some(prev_index_of_prev_block) = prev_block.prev_index {
                self.blocks[prev_index_of_prev_block].next_index = Some(index);
            }

            self.blocks[index].prev_index = prev_block.prev_index;
            self.blocks[index].range.start = prev_block.range.start;
        }

        if let Some(next_free_index) = next_free_index {
            let next_block = self.blocks.remove(next_free_index).unwrap();
            if let Some(next_index_of_next_block) = next_block.next_index {
                self.blocks[next_index_of_next_block].prev_index = Some(index);
            }

            self.blocks[index].next_index = next_block.next_index;
            self.blocks[index].range.end = next_block.range.end;
        }
    }
}
