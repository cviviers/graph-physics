# TODO

> [!NOTE]  
> This is a work in progress to make our work open source and ready to be used by anyone.

## Code wise

- [X] Implement all utils
- [X] Implement all datasets
- [X] Add testing perf dataloaeder with workers
- [X] Find a good way to implement how to compute the node type
- [X] Build the custom functions for aneurysm and co for adding the right attributes
- [X] Test the custom functions
- [X] Implement all layers
- [X] Implement Message Passing and Transformer
- [X] Implement EPD and ETD
- [X] Find a way to not bug if you dont have dgl
- [ ] Implement Simulator
- [ ] Implement train loop with L.Lighting
- [ ] Implement Wandb
- [X] Use proper dataloader
- [ ] Implement proper valdiation metric at the end of each epoch
- [ ] Function to do rendering without paraview (test for 2D and 3D)
- [ ] Implement vizu for one traj at the end of each epoch as well 
- [ ] Add loggers

> [!WARNING]  
> H5-based dataloader does not support multiple workers. XDMF can.

## Colab Wise

- [ ] One notebook for the cylinder
- [ ] One notebook for the coarse aneurysm

## Educational wise

- [ ] Write about how to define the .json and the 2 functions
- [ ] Prepare readme
- [ ] In the readme, say which features we implemented and from what paper

## Dev wise

- [ ] Make setup and requirements
- [ ] Make CI/CD
- [ ] Switch black to ruff