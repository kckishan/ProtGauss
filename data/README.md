# Data Preprocessing

Required steps to follow before running the model:
1. Download UniProt fasta file (release 2019_11) and put it in `fasta` folder.
2. Training sequences are already available as `trainSequence_full.csv` and `trainSequence_eDBD.csv`
3. Create a folder (eg. c_elegans) and place the test sequence fasta file inside that folder.
4. Set the folder name as the value of `organism` variable in line number 63 of `main.py`.

**Folder information**:
- `biovec` saves the embeddings learned by ProtVec to `embeddings` folder.

- `rawdata` folder contains the raw fasta file and binding mode summary files.

- `processedData` stores preprocessed sequences containing the representation for 195 TFs.

Training protvec takes a lot of time and to avoid that we provide pretrained embeddings and the processed data.
You can download them from [Google Drive](https://drive.google.com/drive/folders/1iP1OpCLsbj9lz2ukksQO_p0AMmpwY4i3?usp=sharing).
You must place the downloaded folders inside `data`.