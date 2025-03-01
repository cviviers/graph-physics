python -m graphphysics.train \
            --training_parameters_path=training_config/coarse-aneurysm.json \
            --num_epochs=15 \
            --init_lr=0.001 \
            --batch_size=2 \
            --warmup=1500 \
            --num_workers=2 \
            --prefetch_factor=2 \
            --model_save_path=model.ckpt \
            --no_edge_feature