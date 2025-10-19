
Role:  
You are a senior front‑end engineer and ML instructor building a browser‑only TensorFlow.js MNIST demo for students.

Context:

-   Build a GitHub Pages–deployable web app that TRAINS and RUNS entirely client‑side with TensorFlow.js and tfjs‑vis.
    
-   MNIST data will be provided by the user as two local files via file inputs: mnist_train.csv and mnist_test.csv.
    
-   CSV format: each row = label (0–9) followed by 784 pixel values (0–255) with no header.
    
-   Do NOT fetch data over the network; parse the two uploaded files in the browser, normalize pixels to , reshape to [N,28,28,1], and one‑hot labels to depth 10.[tensorflow](https://www.tensorflow.org/tutorials/load_data/csv?hl=ko)
    
-   Implement FILE‑BASED model Save/Load only: download model.json + weights.bin, and reload from user‑selected files (no IndexedDB/LocalStorage).
    
-   Include training, evaluation with charts, and a test preview that displays 5 random test images in one row with predicted labels.
    

Instruction:  
Output exactly three fenced code blocks, in this order, labeled “index.html”, “data-loader.js”, and “app.js”, implementing all features below without any extra prose.

-   index.html
    
    -   Include CDNs:
        
        -   <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
        -   <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@latest"></script>
    -   Minimal CSS for a two‑column layout and a horizontal preview strip.
        
    -   Controls:
        
        -   “Upload Train CSV” (<input type="file" id="train-csv" accept=".csv">)
            
        -   “Upload Test CSV” (<input type="file" id="test-csv" accept=".csv">)
            
        -   Buttons: Load Data, Train, Evaluate, Test 5 Random, Save Model (Download), Load Model (From Files), Reset, Toggle Visor.
            
        -   Model load inputs: <input type="file" id="upload-json" accept=".json"> and <input type="file" id="upload-weights" accept=".bin">
            
    -   Sections: Data Status, Training Logs, Metrics (overall accuracy + charts), Random 5 Preview (row of canvases + predicted labels), Model Info (layers/params).
        
    -   Defer‑load data-loader.js then app.js.
        
-   data-loader.js
    
    -   Implement file‑based CSV parsing with FileReader/TextDecoder (no external libraries). Robustly handle large files by chunking or streaming if needed; otherwise readAsText is acceptable.
        
    -   Parse rows as: first value → label int, remaining 784 → pixels; ignore empty lines.
        
    -   Normalize pixels /255, reshape to [N,28,28,1], one‑hot labels depth 10.
        
    -   Provide:
        
        -   async function loadTrainFromFiles(file): returns {xs, ys}
            
        -   async function loadTestFromFiles(file): returns {xs, ys}
            
        -   function splitTrainVal(xs, ys, valRatio=0.1): returns {trainXs, trainYs, valXs, valYs}
            
        -   function getRandomTestBatch(xs, ys, k=5): returns tensors for preview
            
        -   function draw28x28ToCanvas(tensor, canvas, scale=4)
            
    -   Dispose intermediate tensors to avoid leaks.
        
-   app.js
    
    -   Wire UI:
        
        -   onLoadData: read both CSV files, build tensors, and show counts.
            
        -   onTrain: build and train the CNN with tfjs‑vis fitCallbacks.
            
        -   onEvaluate: compute test accuracy; render confusion matrix heatmap and per‑class accuracy bar chart in Visor; print overall accuracy.
            
        -   onTestFive: sample 5 random test images, render in one horizontal row; print predicted labels under each (green if correct, red if wrong).
            
        -   onSaveDownload: await model.save('downloads://mnist-cnn')
            
        -   onLoadFromFiles: const m = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile])); replace current model, model.summary(), rebind buttons.
            
        -   onReset: dispose tensors/model and clear UI.
            
    -   Model:
        
        -   tf.sequential([  
            Conv2D(32, 3, activation='relu', padding='same', inputShape:),[tensorflow](https://www.tensorflow.org/tutorials/load_data/csv?hl=ko)  
            Conv2D(64, 3, activation='relu', padding='same'),  
            MaxPool2D(2),  
            Dropout(0.25),  
            Flatten(),  
            Dense(128, activation='relu'),  
            Dropout(0.5),  
            Dense(10, activation='softmax')  
            ]);
            
        -   Compile: optimizer='adam', loss='categoricalCrossentropy', metrics=['accuracy'].
            
        -   Training defaults: epochs 5–10, batchSize 64–128, shuffle true; record duration and best val accuracy.
            
    -   Charts (tfjs‑vis):
        
        -   Live loss/val_loss and acc/val_acc during fit.
            
        -   Confusion matrix and per‑class accuracy on evaluation.
            
    -   Performance & safety:
        
        -   Use tf.tidy where appropriate; dispose old models/tensors on replace.
            
        -   Try/catch around file handling and training; show friendly error messages.
            
        -   Ensure UI stays responsive (await/queueMicrotask, requestAnimationFrame for long operations).
            

Formatting:

-   Produce only three fenced code blocks labeled exactly “index.html”, “data-loader.js”, and “app.js”.
    
-   Browser‑only JavaScript; no Node or extra libraries; clear English comments; no text outside the code blocks.

-   Provide very detail and intuitive comments as explanation for important code blocks.
