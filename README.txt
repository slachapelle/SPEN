Before running the code, you should do the following: 
In main.py, set the LOCAL_FOLDER variable to the name of the folder that will be available to your GPU.

If you are on MILA's servers, you don't need to download or play with the data, everything will be transfer automatically to your LOCAL_FOLDER.

Here's the command to run:

	python main.py experiment GradientDescentPredictor <Your experiment folder> Denoising

The experiment folder will contain every files related to the experiment.

If you want to play with the hyper params, you must change them directly in hyper.py .

CAUTION: The code is fully resumable, so if you run an experiment in folder that already contains the hyper.pkl file, this file will be used as hyperparameters, not those written in the code.

Instead of GradientDescentPredictor, you can also try GDMomentumPredictor, I haven't experiment with it a lot so it might be buggy.

Cheers!

Sebastien
	