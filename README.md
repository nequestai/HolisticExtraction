# Extracting keypoints using Mediapipe Holistic
Script to extract keypoints and save them to both video and a csv with Holistic model results arrays
## macOS Installation
You can skip these 2 steps if you already have Python3 installed. You can verify by using the following commands:
```
which python
python --version
```
### Install homebrew
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
### Install Python3
```
brew install python
```

## Data setup
Copy the mov files from your directory to `/assets` directory at the root of this project

## Running the script
In the terminal, go to the root of this project and run:
```
./extract.sh
```

## Results
Once the script is successfully run, the `/output` directory will have a `.mov` and a `.mov.csv` file corresponding to each file in the `/assets` directory
