### Block Diagram
![alt text](./design_diagrams/complete_block_diagram.png)
- `timestep` repr:-
    - The timestep info, is fed to the `sinosuidal` embedding, which in turn will be fed to `linear layers` and we end up getting the `timestep` representation.
    - This `timestep` repr. is fed to all `Resnet` Blocks i.e `down`, `mid` and `up`.
- `text` repr:-  
    - The `text` i.e `prompt` also called the `controlling signal` is converted to `text repr` after it was fed to a `text encoder`.
    - This `text repr` is attended by  all the `Cross Attention Blocks` of `down`, `mid` and `up` blocks.
- `class` repr:-  
    - The `class repr` that we got after passing it to an `embedding` layer was added to `time repr` and fed to all `Resnet` blocks.


#### More on `TimeStep Repr` and `Class Repr`
![alt text](./design_diagrams/timestep_class_block.png)
- For timestep, the authors use, `TimeEmbeddingLayer`, then `MLP`
- For class repr, they used `learnable` `ClassEmbeddingLayer`




#### Implementation

##### We will implement `DiT-B model`
![alt text](./design_diagrams/dit_b.png)


##### `patch_embedding`
![alt text](./design_diagrams/patch_embedding.png)
- Takes an image and does the following:-
    - Partitions into `patches` with each patch having a `patch_repr` based on `image_height`, `image_width` and `patch_size` from `dit_params`.
    - Creates a sequence of this
    - Bake in the `positional` embedding info, to each `patch_repr`
        ![alt text](./design_diagrams/pos_patch_emb.png)
        - ViT: 1D pos repr
        - DiT: 2D pos repr 



