<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <h1>YOLO Latest version Integration with Label Studio Backend</h1>
    <p>This guide provides instructions on integrating the YOLO model with Label Studio using the Label Studio ML backend. It is recommended to use WSL (Windows Subsystem for Linux) or a Linux environment to avoid issues with the fork package.</p>
    
<h2>Prerequisites</h2>
<ul>
    <li>Windows with WSL installed or a Linux environment</li>
    <li>Python 3.6 or higher</li>
    <li>Redis server</li>
    <li>Label Studio</li>
</ul>

<h2>Installation Steps</h2>

<h3>1. Install Required Packages</h3>
<p>First, install the necessary Python packages:</p>
<pre><code>pip install ultralytics redis rq</code></pre>

<h3>2. Install Label Studio</h3>
<p>Upgrade and install Label Studio:</p>
<pre><code>pip install -U label-studio</code></pre>

<h3>3. Install Label Studio Backend</h3>
<p>Follow the guide provided on the Label Studio website to set up the ML backend: 
    <a href="https://labelstud.io/guide/ml_create" target="_blank">Label Studio ML Backend Guide</a>
</p>

<h3>4. Integrate the Backend with Label Studio</h3>
<p>To integrate the YOLO model backend with Label Studio, follow these steps:</p>

<h4>I. Install WSL (if using Windows)</h4>
<p>For a detailed guide on installing WSL, visit the official Microsoft documentation: 
    <a href="https://learn.microsoft.com/en-us/windows/wsl/install" target="_blank">Install WSL</a>
</p>
<p>After setting up WSL, install the necessary packages:</p>
<pre><code>pip install python ultralytics torch redis rq</code></pre>
<p>If you encounter any errors, refer to the <a href="https://pypi.org/" target="_blank">PyPI documentation</a> for troubleshooting.</p>

<h4>II. Start the Redis Server</h4>
<p>Open an Ubuntu terminal and start the Redis server:</p>
<pre><code>redis-server</code></pre>

<h4>III. Run the Worker</h4>
<p>Open another Ubuntu terminal, navigate to your YOLO backend directory, and run the worker:</p>
<pre><code>cd my_ml_backyolotest
python3 worker_rq.py</code></pre>

<h4>IV. Start the ML Backend</h4>
<p>Open another Ubuntu terminal and start the ML backend:</p>
<pre><code>label-studio-ml start my_ml_backyolotest</code></pre>

<h2>Using Label Studio</h2>
<ol>
    <li>Open your web browser and navigate to <a href="http://localhost:8080" target="_blank">http://localhost:8080</a>.</li>
    <li>Create a new project and select the Object Detection template.</li>
    <li>Upload your images for annotation.</li>
    <li>Connect the YOLO model to your project:
        <ul>
            <li>Go to the <strong>Settings</strong> in the top navigation bar.</li>
            <li>Find the <strong>Model</strong> section in the left sidebar.</li>
            <li>Connect your model by following the prompts.</li>
        </ul>
    </li>
</ol>
<p>You are now ready to use YOLO for object detection in Label Studio!</p>

<p>This README provides a structured guide for setting up and integrating YOLO with Label Studio. If you encounter any issues, refer to the respective documentation for detailed troubleshooting steps.</p>
</body>
</html>

 

