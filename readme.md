# LSST Satellite streak analysis

This repo contains the code used for the analysis of the paper...

## Usage
1. Make sure all the necessary packages listed in requirements.txt are installed.

2. Download the LSST scheduler from https://s3df.slac.stanford.edu/data/rubin/sim-data/sims_featureScheduler_runs3.2/baseline/baseline_v3.2_10yrs.db and place it into the data folder

3. Run the script lsst_observations.py in the data folder.

4. Run the streaks.py script in the constellation folder. 
    * Further usage instructions are at the top of the script

5. Run a satellite model analysis using the notebooks in the analysis directory.
    * The V2 and V1.5 models are included. 
   
7. Run a custom satellite model(optional)
    * If you have a satellite model built from LUMOS, copy one of the analysis notebooks.
    * Replace the default model with your own model.

