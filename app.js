class MNISTApp {
    constructor() {
        this.dataLoader = new MNISTDataLoader();
        this.model = null;
        this.isTraining = false;
        this.trainData = null;
        this.testData = null;
        
        this.initializeUI();
    }

    initializeUI() {
        // Bind button events
        document.getElementById('loadDataBtn').addEventListener('click', () => this.onLoadData());
        document.getElementById('trainBtn').addEventListener('click', () => this.onTrain());
        document.getElementById('evaluateBtn').addEventListener('click', () => this.onEvaluate());
        document.getElementById('testFiveBtn').addEventListener('click', () => this.onTestFive());
        document.getElementById('saveModelBtn').addEventListener('click', () => this.onSaveDownload());
        document.getElementById('loadModelBtn').addEventListener('click', () => this.onLoadFromFiles());
        document.getElementById('resetBtn').addEventListener('click', () => this.onReset());
        document.getElementById('toggleVisorBtn').addEventListener('click', () => this.toggleVisor());
    }

    async onLoadData() {
        try {
            const trainFile = document.getElementById('trainFile').files[0];
            const testFile = document.getElementById('testFile').files[0];
            
            if (!trainFile || !testFile) {
                this.showError('Please select both train and test CSV files');
                return;
            }

            this.showStatus('Loading training data...');
            const trainData = await this.dataLoader.loadTrainFromFiles(trainFile);
            
            this.showStatus('Loading test data...');
            const testData = await this.dataLoader.loadTestFromFiles(testFile);

            this.trainData = trainData;
            this.testData = testData;

            this.updateDataStatus(trainData.count, testData.count);
            
            // Проверяем данные
            this.showStatus('✓ Data loaded successfully!');
            this.showStatus(`Data check - Train shape: ${trainData.xs.shape}, Test shape: ${testData.xs.shape}`);
            this.showStatus(`Data range - Min: ${trainData.xs.min().dataSync()[0].toFixed(3)}, Max: ${trainData.xs.max().dataSync()[0].toFixed(3)}`);
            
        } catch (error) {
            this.showError(`Failed to load data: ${error.message}`);
        }
    }

    async onTrain() {
        if (!this.trainData) {
            this.showError('Please load training data first');
            return;
        }

        if (this.isTraining) {
            this.showError('Training already in progress');
            return;
        }

        try {
            this.isTraining = true;
            
            this.showStatus('=== DEBUG MODE: Creating SIMPLE working autoencoder ===');
            
            // Создаем максимально простую модель
            this.model = this.createSimpleWorkingAutoencoder();
            this.updateModelInfo();
            
            // ТЕСТ 1: Проверяем архитектуру
            this.showStatus('Testing model architecture...');
            const testInput = this.trainData.xs.slice([0, 0, 0, 0], [1, 28, 28, 1]);
            const testOutput = this.model.predict(testInput);
            this.showStatus(`✓ Model test: Input ${testInput.shape} → Output ${testOutput.shape}`);
            
            // ТЕСТ 2: Проверяем значения
            const outputData = await testOutput.data();
            this.showStatus(`✓ Output range - Min: ${Math.min(...outputData).toFixed(3)}, Max: ${Math.max(...outputData).toFixed(3)}`);
            
            testInput.dispose();
            testOutput.dispose();

            this.showStatus('=== Starting REAL training ===');
            
            // Берем маленький набор для быстрого теста
            const trainSubset = this.trainData.xs.slice([0, 0, 0, 0], [500, 28, 28, 1]);
            const noisyData = this.dataLoader.addNoise(trainSubset, 0.1);

            // Проверяем зашумленные данные
            const noisyMin = (await noisyData.min().data())[0];
            const noisyMax = (await noisyData.max().data())[0];
            this.showStatus(`Noisy data range: ${noisyMin.toFixed(3)} to ${noisyMax.toFixed(3)}`);

            const startTime = Date.now();
            const history = await this.model.fit(noisyData, trainSubset, {
                epochs: 5, // Всего 5 эпох для быстрого теста
                batchSize: 16,
                validationSplit: 0.2,
                shuffle: true,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        this.showStatus(`Epoch ${epoch + 1}/5 - loss: ${logs.loss.toFixed(6)}, val_loss: ${logs.val_loss.toFixed(6)}`);
                        
                        // Быстрый тест после каждой эпохи
                        if (epoch % 2 === 0) {
                            setTimeout(() => this.quickTest(), 100);
                        }
                    }
                }
            });

            const duration = (Date.now() - startTime) / 1000;
            const finalLoss = history.history.loss[history.history.loss.length - 1];
            
            this.showStatus(`=== Training completed in ${duration.toFixed(1)}s ===`);
            this.showStatus(`✓ Final loss: ${finalLoss.toFixed(6)}`);
            
            if (finalLoss < 0.01) {
                this.showStatus('🎉 SUCCESS: Model is learning!');
            } else {
                this.showStatus('⚠️ Model might need more training');
            }
            
            // Очистка
            trainSubset.dispose();
            noisyData.dispose();
            
            // Финальный тест
            await this.onTestFive();
            
        } catch (error) {
            this.showError(`Training failed: ${error.message}`);
            console.error('Training error:', error);
        } finally {
            this.isTraining = false;
        }
    }

    async quickTest() {
        if (!this.model || !this.testData) return;
        
        try {
            const { batchXs, batchYs, indices } = this.dataLoader.getRandomTestBatch(
                this.testData.xs, this.testData.ys, 2
            );
            
            const noisyBatchXs = this.dataLoader.addNoise(batchXs, 0.1);
            const reconstructions = this.model.predict(noisyBatchXs);
            
            // Проверяем выходные значения
            const reconData = await reconstructions.data();
            const avgOutput = reconData.reduce((a, b) => a + b) / reconData.length;
            this.showStatus(`Quick test - Avg output: ${avgOutput.toFixed(3)}`);
            
            noisyBatchXs.dispose();
            reconstructions.dispose();
            batchXs.dispose();
            batchYs.dispose();
            
        } catch (error) {
            console.log('Quick test failed:', error.message);
        }
    }

    async onEvaluate() {
        if (!this.model) {
            this.showError('No model available. Please train or load a model first.');
            return;
        }

        if (!this.testData) {
            this.showError('No test data available');
            return;
        }

        try {
            this.showStatus('Evaluating model performance...');
            
            const testSubset = this.testData.xs.slice([0, 0, 0, 0], [100, 28, 28, 1]);
            const noisyData = this.dataLoader.addNoise(testSubset, 0.1);
            
            const reconstructions = this.model.predict(noisyData);
            
            // Детальная диагностика
            const originalData = await testSubset.data();
            const reconData = await reconstructions.data();
            
            const mse = reconstructions.sub(testSubset).square().mean();
            const mseValue = (await mse.data())[0];
            
            const avgOriginal = originalData.reduce((a, b) => a + b) / originalData.length;
            const avgRecon = reconData.reduce((a, b) => a + b) / reconData.length;
            
            this.showStatus(`=== Detailed Analysis ===`);
            this.showStatus(`Original avg: ${avgOriginal.toFixed(3)}, Reconstructed avg: ${avgRecon.toFixed(3)}`);
            this.showStatus(`MSE: ${mseValue.toFixed(6)}`);
            
            if (mseValue < 0.01) {
                this.showStatus('✅ EXCELLENT: Model is working well!');
            } else if (mseValue < 0.05) {
                this.showStatus('⚠️ OK: Model needs more training');
            } else {
                this.showStatus('❌ POOR: Model is not learning properly');
            }
            
            testSubset.dispose();
            noisyData.dispose();
            reconstructions.dispose();
            mse.dispose();
            
        } catch (error) {
            this.showError(`Evaluation failed: ${error.message}`);
        }
    }

    async onTestFive() {
        if (!this.model || !this.testData) {
            this.showError('Please load both model and test data first');
            return;
        }

        try {
            this.showStatus('Running final denoising test...');
            
            const { batchXs, batchYs, indices } = this.dataLoader.getRandomTestBatch(
                this.testData.xs, this.testData.ys, 5
            );
            
            const noisyBatchXs = this.dataLoader.addNoise(batchXs, 0.1);
            const reconstructions = this.model.predict(noisyBatchXs);
            
            // Проверяем выход перед отрисовкой
            const reconData = await reconstructions.data();
            const hasValidOutput = reconData.some(val => val > 0.1 && val < 0.9);
            
            if (!hasValidOutput) {
                this.showStatus('❌ WARNING: Model output is not valid (all values too extreme)');
            }
            
            const trueLabels = batchYs.argMax(-1);
            const trueArray = await trueLabels.array();
            
            this.renderDenoisingPreview(batchXs, noisyBatchXs, reconstructions, trueArray, indices);
            
            noisyBatchXs.dispose();
            reconstructions.dispose();
            batchXs.dispose();
            batchYs.dispose();
            trueLabels.dispose();
            
        } catch (error) {
            this.showError(`Denoising test failed: ${error.message}`);
        }
    }

    async onSaveDownload() {
        if (!this.model) {
            this.showError('No model to save');
            return;
        }

        try {
            await this.model.save('downloads://mnist-working-autoencoder');
            this.showStatus('Model saved successfully!');
        } catch (error) {
            this.showError(`Failed to save model: ${error.message}`);
        }
    }

    async onLoadFromFiles() {
        const jsonFile = document.getElementById('modelJsonFile').files[0];
        const weightsFile = document.getElementById('modelWeightsFile').files[0];
        
        if (!jsonFile || !weightsFile) {
            this.showError('Please select both model.json and weights.bin files');
            return;
        }

        try {
            this.showStatus('Loading model...');
            
            if (this.model) {
                this.model.dispose();
            }
            
            this.model = await tf.loadLayersModel(
                tf.io.browserFiles([jsonFile, weightsFile])
            );
            
            this.updateModelInfo();
            this.showStatus('Model loaded successfully!');
            
        } catch (error) {
            this.showError(`Failed to load model: ${error.message}`);
        }
    }

    onReset() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
        
        this.dataLoader.dispose();
        this.trainData = null;
        this.testData = null;
        
        this.updateDataStatus(0, 0);
        this.updateModelInfo();
        this.clearPreview();
        this.showStatus('Reset completed');
    }

    toggleVisor() {
        tfvis.visor().toggle();
    }

    createSimpleWorkingAutoencoder() {
        const model = tf.sequential();
        
        // САМАЯ ПРОСТАЯ РАБОТАЮЩАЯ АРХИТЕКТУРА
        // Всего 3 слоя - вход, скрытый, выход
        model.add(tf.layers.conv2d({
            inputShape: [28, 28, 1],
            filters: 4,  // Минимум фильтров
            kernelSize: 3,
            activation: 'relu',
            padding: 'same',
            name: 'conv1'
        }));
        
        model.add(tf.layers.conv2d({
            filters: 4,
            kernelSize: 3,
            activation: 'relu', 
            padding: 'same',
            name: 'conv2'
        }));
        
        model.add(tf.layers.conv2d({
            filters: 1,
            kernelSize: 3,
            activation: 'sigmoid', // SIGMOID для выхода 0-1
            padding: 'same',
            name: 'output'
        }));
        
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError',
            metrics: ['mse']
        });
        
        console.log('✓ Simple working autoencoder created');
        model.summary();
        
        return model;
    }

    renderDenoisingPreview(original, noisy, reconstructed, trueLabels, indices) {
        const container = document.getElementById('previewContainer');
        container.innerHTML = '<h3>Denoising Test Results:</h3>';
        
        const originalArray = original.arraySync();
        const noisyArray = noisy.arraySync();
        const reconstructedArray = reconstructed.arraySync();
        
        for (let i = 0; i < originalArray.length; i++) {
            const group = document.createElement('div');
            group.className = 'preview-group';
            group.style.cssText = 'border: 2px solid #666; padding: 10px; margin: 10px 0; background: #f5f5f5;';
            
            const title = document.createElement('div');
            title.textContent = `Sample ${indices[i]} - Digit: ${trueLabels[i]}`;
            title.style.cssText = 'font-weight: bold; margin-bottom: 8px; color: #333;';
            group.appendChild(title);
            
            const debugInfo = document.createElement('div');
            const origAvg = originalArray[i].reduce((a, b) => a + b) / originalArray[i].length;
            const reconAvg = reconstructedArray[i].reduce((a, b) => a + b) / reconstructedArray[i].length;
            debugInfo.textContent = `Debug - Orig avg: ${origAvg.toFixed(3)}, Recon avg: ${reconAvg.toFixed(3)}`;
            debugInfo.style.cssText = 'font-size: 12px; color: #666; margin-bottom: 8px;';
            group.appendChild(debugInfo);
            
            const row = document.createElement('div');
            row.className = 'preview-row';
            row.style.cssText = 'display: flex; justify-content: space-around;';
            
            const items = [
                { data: originalArray[i], label: 'Original' },
                { data: noisyArray[i], label: 'Noisy (10%)' },
                { data: reconstructedArray[i], label: 'Reconstructed' }
            ];
            
            items.forEach(item => {
                const itemDiv = document.createElement('div');
                itemDiv.className = 'preview-item';
                itemDiv.style.cssText = 'text-align: center;';
                
                const label = document.createElement('div');
                label.textContent = item.label;
                label.style.cssText = 'margin-bottom: 5px; font-weight: bold;';
                
                const canvas = document.createElement('canvas');
                this.dataLoader.draw28x28ToCanvas(tf.tensor(item.data), canvas, 6);
                
                itemDiv.appendChild(label);
                itemDiv.appendChild(canvas);
                row.appendChild(itemDiv);
            });
            
            group.appendChild(row);
            container.appendChild(group);
        }
    }

    clearPreview() {
        document.getElementById('previewContainer').innerHTML = '';
    }

    updateDataStatus(trainCount, testCount) {
        const statusEl = document.getElementById('dataStatus');
        statusEl.innerHTML = `
            <h3>Data Status</h3>
            <p>Train samples: ${trainCount}</p>
            <p>Test samples: ${testCount}</p>
        `;
    }

    updateModelInfo() {
        const infoEl = document.getElementById('modelInfo');
        
        if (!this.model) {
            infoEl.innerHTML = '<h3>Model Info</h3><p>No model loaded</p>';
            return;
        }
        
        let totalParams = 0;
        this.model.layers.forEach(layer => {
            layer.getWeights().forEach(weight => {
                totalParams += weight.size;
            });
        });
        
        infoEl.innerHTML = `
            <h3>Model Info</h3>
            <p>Layers: ${this.model.layers.length}</p>
            <p>Parameters: ${totalParams.toLocaleString()}</p>
            <p style="color: orange; font-weight: bold;">✓ Simple Debug Model</p>
        `;
    }

    showStatus(message) {
        const logs = document.getElementById('trainingLogs');
        const entry = document.createElement('div');
        entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        logs.appendChild(entry);
        logs.scrollTop = logs.scrollHeight;
    }

    showError(message) {
        this.showStatus(`❌ ERROR: ${message}`);
        console.error(message);
    }
}

// Initialize app when page loads
document.addEventListener('DOMContentLoaded', () => {
    new MNISTApp();
});
