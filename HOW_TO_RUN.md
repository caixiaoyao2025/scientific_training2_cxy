# How to Run the Bioinformatics Tool-Demo Agent

## Quick Start

### 1. Open the Colab Notebook

Download `colab_demo.ipynb` from this repository and open it in Google Colab.

### 2. Run the Setup Cell

Run the second cell and wait for the initialization to finish. This will:
- Install Python 3.11 and bioinformatics tools (fastp, samtools, bedtools)
- Clone the repository and set up the environment
- Launch the Gradio interface

### 3. Open the Gradio Interface

Find the line:
```
Running on public URL: https://xxxx.gradio.live
```
Click on the link to open the web interface.

### 4. Enter Your Prompt

Find the input area at the bottom of the webpage and enter your prompt.

**Example prompt:**
```
Download these files from GitHub and run the analysis:
1. Download https://raw.githubusercontent.com/caixiaoyao2025/scientific_training2_cxy/main/data/sample.fastq and run fastp on it
2. Download https://raw.githubusercontent.com/caixiaoyao2025/scientific_training2_cxy/main/data/sample.sam and run samtools flagstat on it
3. Download https://raw.githubusercontent.com/caixiaoyao2025/scientific_training2_cxy/main/data/sample.bed and https://raw.githubusercontent.com/caixiaoyao2025/scientific_training2_cxy/main/data/sample_annotation.gff, then use bedtools intersect to find overlaps
Summarize all results.
```

### 5. Wait for Results

The agent will execute each step and display results in real time.

## Tips

1. **Be patient with rate limits**: Due to the rate limit of requests, please be patient if the execution log does not refresh for a few minutes.

2. **Don't panic on errors**: Don't be hurried to kill the program when you see "error" in the execution log. The LLM may deal with it automatically.

## Screen Recording

<video src="屏幕录制%202026-07-23%20212451.mp4" controls width="640"></video>
