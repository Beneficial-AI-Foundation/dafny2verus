use vstd::prelude::*;

verus! {
    struct BoundedQueue<T> where T: Copy {
        // abstract state
        ghost contents: Seq<T>, // the contents of the bounded queue
        ghost N: nat, // the (maximum) size of the bounded queue
        
        // concrete state
        data: Vec<T>,
        wr: usize,
        rd: usize,
    }

    impl<T> BoundedQueue<T> where T: Copy {
        spec fn valid(self) -> bool {
            self.data.len() == self.N + 1 &&
            self.wr <= self.N &&
            self.rd <= self.N &&
            self.contents == if self.rd <= self.wr {
                self.data@.subrange(self.rd as int, self.wr as int)
            } else {
                self.data@.subrange(self.rd as int, self.data.len() as int) + 
                self.data@.subrange(0, self.wr as int)
            }
        }

        fn insert(&mut self, x: T)
            requires 
                old(self).valid(),
                old(self).contents.len() != old(self).N,
            ensures 
                self.contents == old(self).contents.push(x),
                self.N == old(self).N,
                self.valid(),
        {
            proof {
                self.contents = self.contents.push(x);
            }
            
            self.data.set(self.wr, x);
            
            // Helper assertions for verification
            assert(self.wr == self.data.len() - 1 ==> 
                   self.contents == if self.rd <= 0 { 
                       self.data@.subrange(self.rd as int, 0) 
                   } else { 
                       self.data@.subrange(self.rd as int, self.data.len() as int) + 
                       self.data@.subrange(0, 0) 
                   });
            
            assert(self.wr != self.data.len() - 1 ==> 
                   self.contents == if self.rd <= self.wr + 1 { 
                       self.data@.subrange(self.rd as int, (self.wr + 1) as int) 
                   } else { 
                       self.data@.subrange(self.rd as int, self.data.len() as int) + 
                       self.data@.subrange(0, (self.wr + 1) as int) 
                   });
            
            if self.wr == self.data.len() - 1 {
                self.wr = 0;
            } else {
                self.wr = self.wr + 1;
            }
            
            assert(self.contents == if self.rd <= self.wr {
                self.data@.subrange(self.rd as int, self.wr as int)
            } else {
                self.data@.subrange(self.rd as int, self.data.len() as int) + 
                self.data@.subrange(0, self.wr as int)
            });
        }

        fn remove(&mut self) -> (x: T)
            requires 
                old(self).valid(),
                old(self).contents.len() != 0,
            ensures 
                self.contents == old(self).contents.drop_first(),
                old(self).contents.index(0) == x,
                self.N == old(self).N,
                self.valid(),
        {
            proof {
                self.contents = self.contents.drop_first();
            }
            
            let x = self.data[self.rd];
            
            assert(self.rd < self.data.len());
            assert(self.rd <= self.N);
            
            if self.rd == self.data.len() - 1 {
                self.rd = 0;
            } else {
                self.rd = self.rd + 1;
            }
            
            assert(self.rd <= self.N);
            assert(self.contents == if self.rd <= self.wr {
                self.data@.subrange(self.rd as int, self.wr as int)
            } else {
                self.data@.subrange(self.rd as int, self.data.len() as int) + 
                self.data@.subrange(0, self.wr as int)
            });
            
            x
        }
    }
}