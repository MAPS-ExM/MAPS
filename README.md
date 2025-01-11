# Mitochondrial Automated Pan-ExM Segmentation

# Noisy Antibodies 
![KidneyWorkflow](./docs/KidneyWorkflow.png)

If your data does not allow to derive prediction targets masks in a simple way (like morphological oprations), then this is the case to consider. This corresponds to the analysis of the kidney tissue in the paper and the workflow is illustrated in the figure above. We first train a model to predict the mitochondria outline from the antibodies before fine-tuning the model to the inner structure. The code for this workflow is contained in `NoisyImmunolabeling`

### 1. Mitochondria outline
What to do when new data arrives:
1. Test if the data is similar enough so that the old model just works:
    1. Run `NoisyImmunolabelin/Predict_refined_model.py`
    - If yes, perfect, nothing to be done.
    - If no, we have to retrain following the next steps
2. Run some experiments with `NoisyImmunolabeling/train_outline.py` with a small (~3) number of files to find the appropriate lower threshold for the immunolabelling/ antibody (AB) value. Visual inspection gives a first clue and finding the proper value can normally be achieved with less than 5 attempts. The methodolgy is described in the paper.
3. Use this model to get the mitochondria outline for the new batch (automatically created with `train_outline.py`) or for other data with `prediction_outline.py`.

If everything works out fine, you should be able to predict the outline of the target organellese (mitochondria in this example) fairly accurately:
![OutlineExample](./docs/OutlineExample.png)
Depending on the localisation and accuracy of your immunolabelling you might have to experiment with the AB-threshold value or the $\tau$ parameter which specifies the radius of the neighbourhood size [shown as 'Antibody Mask' in the first figure, see the paper for details].



### 2. Inner Structure
1. Run the HeLa cell model or any other available on the kidney data but only output matrix vs cristae dicision and apply Majority Vote if an Ensemble was used. Use these predictions only for the outline predicted by the model from step 1.
2. Use `train_initial_inner.py` to train an initial model that can replicate the combination of HeLa cell prediction and the mitochondria outline. The advantage of this step is that we can basically use all available data for this step.
3. If we want the overall cristae segmentation: `GenerateSegData.ipynb` was used to combine the outline of the parent directory model with the HeLa cell model. This data was then manueally refined and is saved in `/data`.
4. If we want to differentiate between stripped and non stripped:

# Precise Additional Dye - MitoTracker
This case corresponds to the scenario in which an additional dye like MitoTracker allows to automatically derive segmentation masks that work as the prediction targets. 
![MitoTracker](./docs/MitoTracker.png)

If the prediction target can be outlined well enough by the additional dye, any segmentation framework could be used in general like the excellent [nnUNet](https://github.com/MIC-DKFZ/nnUNet). 
The work presented in our paper however only uses the MitoTracker for the general outline of the structure and then fine-tunes an additional encoder to segment the inner structure.
The code for this workflow is found in the `MitoTracker` directory.



## Errors:
If you run the scripts like `train_outline.py` from within their directories, update the `PYTHONPATH` variable like
`export PYTHONPATH='YOURPATH/src/MAPS/':$PYTHONPATH` to make sure all imports work.