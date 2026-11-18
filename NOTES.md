# Notes

My own explanations of the concepts behind this project.

## Compilers and IR
**What are the frontend, middle-end and backend of a compiler? Why does this project stop at AIR?**


**What is LLVM IR, and what does SSA mean? Why does SSA help?**


**What is AIR?**


## Optimization
**What changed between `-O0` and optimized `saxpy`, and which pass did it?**


**What does `getelementptr` do? What do `addrspace(1)`, `addrspace(2)` and `addrspace(3)` mean?**


**What does the `fast` flag on `fadd`/`fmul` allow?**


**Why does `matmul` gain `phi` instructions after optimization?**


## Representation
**Why do ML-guided compilers need program representations?**


**How does IR2Vec work, and how is my representation simpler?**


**Why normalize the counts? What are the limitations of counting instructions?**


## Analysis
**What do cosine similarity and PCA tell me?**


**Which kernels clustered, and why?**


**Why doesn't `histogram` stand apart even though it uses atomics?**


**Why were `matmul` and `blur3` the least consistent under optimization?**


## GPU concepts
**Why does `reduce_sum` need barriers? Why must the add in `histogram` be atomic?**

