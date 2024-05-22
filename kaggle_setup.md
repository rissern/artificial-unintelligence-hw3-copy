# Kaggle Setup
In order to participate in the kaggle challenge, you must select your best model and hyperparameters and run the forward pass on the test dataset. The test dataset is formatted in the same manner as your `Train` dataset. Note, the `groundTruth.tif` files are dummies and should not be used for training. They are simply there to enable you to utilized the code you've developed without major changes.

In order to evaluate your best model with the test images, you will need to adapt `src/esd_data/datamodule`.

In `prepare_data` You will need to replace the previous `train_test_split` single line code to the following code:
```
if self.train_size == 1:
    tile_dirs_train, tile_dirs_val = tile_dirs, []
else:
    tile_dirs_train, tile_dirs_val = train_test_split(
                    tile_dirs,
                    test_size=1 - self.train_size,
                    random_state=self.seed
    )
```
In `setup` you will need to define a `test` stage.
```
if stage == "test":
    self.test_dataset = ESDDataset(
    self.train_dir,
    transform=self.transform,
    satellite_type_list=self.satellite_type_list,
    slice_size=self.slice_size,
    )
```
You will also  need to define a `test_dataloader`.

Next run your best model on the test dataset, and post-process the predictions to be in kaggle csv format by running `scripts/evaluate_kaggle.py`.
