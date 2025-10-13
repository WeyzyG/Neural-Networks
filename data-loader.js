class MNISTDataLoader {
    constructor() {
        this.trainData = null;
        this.testData = null;
    }

    // Parse CSV file and convert to tensors
    async loadCSVFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (event) => {
                try {
                    const content = event.target.result;
                    const lines = content.split('\n').filter(line => line.trim() !== '');
                    
                    const labels = [];
                    const pixels = [];
                    
                    for (const line of lines) {
                        const values = line.split(',').map(Number);
                        if (values.length !== 785) continue; // label + 784 pixels
                        
                        labels.push(values[0]);
                        pixels.push(values.slice(1));
                    }
                    
                    if (labels.length === 0) {
                        reject(new Error('No valid data found in file'));
                        return;
                    }
                    
                    // Normalize pixels to [0, 1] and reshape to [N, 28, 28, 1]
                    const xs = tf.tidy(() => {
                        return tf.tensor2d(pixels)
                            .div(255)
                            .reshape([labels.length, 28, 28, 1]);
                    });
                    
                    // One-hot encode labels (still useful for reference)
                    const ys = tf.tidy(() => {
                        return tf.oneHot(labels, 10);
                    });
                    
                    resolve({ xs, ys, count: labels.length, labels });
                } catch (error) {
                    reject(error);
                }
            };
            
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    async loadTrainFromFiles(file) {
        this.trainData = await this.loadCSVFile(file);
        return this.trainData;
    }

    async loadTestFromFiles(file) {
        this.testData = await this.loadCSVFile(file);
        return this.testData;
    }

    // Add Gaussian noise to images
    addNoise(images, noiseStd = 0.5) {
        return tf.tidy(() => {
            const noise = tf.randomNormal(images.shape, 0, noiseStd);
            const noisyImages = images.add(noise).clipByValue(0, 1);
            return noisyImages;
        });
    }

    // Add noise to flattened images (for dense network)
    addNoiseDense(images, noiseStd = 0.5) {
        return tf.tidy(() => {
            const noise = tf.randomNormal(images.shape, 0, noiseStd);
            const noisyImages = images.add(noise).clipByValue(0, 1);
            return noisyImages;
        });
    }

    // Split training data into train/validation sets
    splitTrainVal(xs, ys, valRatio = 0.1) {
        return tf.tidy(() => {
            const numVal = Math.floor(xs.shape[0] * valRatio);
            const numTrain = xs.shape[0] - numVal;
            
            const trainXs = xs.slice([0, 0, 0, 0], [numTrain, 28, 28, 1]);
            const trainYs = ys.slice([0, 0], [numTrain, 10]);
            
            const valXs = xs.slice([numTrain, 0, 0, 0], [numVal, 28, 28, 1]);
            const valYs = ys.slice([numTrain, 0], [numVal, 10]);
            
            return { trainXs, trainYs, valXs, valYs };
        });
    }

    // Get random batch for preview - modified for denoising
    getRandomTestBatch(xs, ys, k = 5) {
        return tf.tidy(() => {
            const shuffledIndices = tf.util.createShuffledIndices(xs.shape[0]);
            const selectedIndices = Array.from(shuffledIndices.slice(0, k));
            
            const batchXs = tf.gather(xs, selectedIndices);
            const batchYs = tf.gather(ys, selectedIndices);
            
            return { batchXs, batchYs, indices: selectedIndices };
        });
    }

    // Draw 28x28 tensor to canvas
    draw28x28ToCanvas(tensor, canvas, scale = 4) {
        return tf.tidy(() => {
            const ctx = canvas.getContext('2d');
            const imageData = new ImageData(28, 28);
            
            // Ensure tensor is 2D and denormalize
            const data = tensor.reshape([28, 28]).mul(255).dataSync();
            
            for (let i = 0; i < 784; i++) {
                const val = data[i];
                imageData.data[i * 4] = val;     // R
                imageData.data[i * 4 + 1] = val; // G
                imageData.data[i * 4 + 2] = val; // B
                imageData.data[i * 4 + 3] = 255; // A
            }
            
            // Scale up for better visibility
            canvas.width = 28 * scale;
            canvas.height = 28 * scale;
            ctx.imageSmoothingEnabled = false;
            
            // Create temporary canvas for scaling
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 28;
            tempCanvas.height = 28;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.putImageData(imageData, 0, 0);
            
            ctx.drawImage(tempCanvas, 0, 0, 28 * scale, 28 * scale);
        });
    }

    // Calculate PSNR (Peak Signal-to-Noise Ratio)
    calculatePSNR(original, reconstructed) {
        return tf.tidy(() => {
            const mse = reconstructed.sub(original).square().mean();
            const psnr = tf.scalar(20).mul(tf.scalar(1).log().div(tf.scalar(10).log())).sub(mse.log().div(tf.scalar(10).log()).mul(tf.scalar(10)));
            return psnr;
        });
    }

    // Clean up stored data
    dispose() {
        if (this.trainData) {
            this.trainData.xs.dispose();
            this.trainData.ys.dispose();
            this.trainData = null;
        }
        if (this.testData) {
            this.testData.xs.dispose();
            this.testData.ys.dispose();
            this.testData = null;
        }
    }
}
