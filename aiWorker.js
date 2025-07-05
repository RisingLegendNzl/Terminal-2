// aiWorker.js - Web Worker for TensorFlow.js AI Model (Multi-Output)

// Import TensorFlow.js library within the worker
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs');

// TensorFlow.js specific configurations
const TFJS_MODEL_STORAGE_KEY = 'roulette-ml-model';
const SEQUENCE_LENGTH = 5;
const LSTM_UNITS = 32;
const EPOCHS = 50;
const BATCH_SIZE = 16;
const TRAINING_MIN_HISTORY = 10;

// Define failure modes for consistency
const failureModes = ['none', 'normalLoss', 'streakBreak', 'sectionShift'];

let mlModel = null;
let scaler = null;
let allPredictionTypes = [];
let terminalMapping = {};
let rouletteWheel = [];

let isTraining = false;

// Helper to get number properties
function getNumberProperties(num) {
    const redNumbers = [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36];
    const getRouletteNumberColor = (number) => {
        if (number === 0) return 'green';
        if (redNumbers.includes(number)) return 'red';
        return 'black';
    };
    const color = getRouletteNumberColor(num);

    const isEven = num % 2 === 0 && num !== 0;
    const isOdd = num % 2 !== 0;
    const isHigh = num >= 19 && num <= 36;
    const isLow = num >= 1 && num <= 18;
    const isZero = num === 0;

    const isD1 = num >= 1 && num <= 12; // First dozen
    const isD2 = num >= 13 && num <= 24; // Second dozen
    const isD3 = num >= 25 && num <= 36; // Third dozen

    // Columns (based on standard layout: 1,4,7...; 2,5,8...; 3,6,9...)
    const isCol1 = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34].includes(num);
    const isCol2 = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35].includes(num);
    const isCol3 = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36].includes(num);


    return {
        isEven: isEven,
        isOdd: isOdd,
        isRed: color === 'red',
        isBlack: color === 'black',
        isGreen: color === 'green',
        isHigh: isHigh,
        isLow: isLow,
        isZero: isZero,
        isD1: isD1,
        isD2: isD2,
        isD3: isD3,
        isCol1: isCol1,
        isCol2: isCol2,
        isCol3: isCol3,
    };
}

// Function to prepare data for the multi-output LSTM training
function prepareDataForLSTM(historyData) {
    const validHistory = historyData.filter(item => item.status === 'success' && item.winningNumber !== null);
    if (validHistory.length < SEQUENCE_LENGTH + 1) {
        self.postMessage({ type: 'status', message: `AI Model: Need at least ${TRAINING_MIN_HISTORY} confirmed spins to train.` });
        return { xs: null, ys: null, featureCount: 0, groupLabelCount: 0, failureLabelCount: 0 };
    }

    const getFeatures = (item) => {
        const properties = getNumberProperties(item.winningNumber);
        const features = [
            item.num1 / 36,
            item.num2 / 36,
            item.difference / 36,
            (item.num1 + item.num2) / 72,
            item.pocketDistance !== null ? item.pocketDistance / 18 : 0,
            // Normalized pocket distance for the *recommended* group
            item.recommendedGroupPocketDistance !== null ? item.recommendedGroupPocketDistance / 18 : 1,
            properties.isEven ? 1 : 0,
            properties.isOdd ? 1 : 0,
            properties.isRed ? 1 : 0,
            properties.isBlack ? 1 : 0,
            properties.isGreen ? 1 : 0,
            properties.isHigh ? 1 : 0,
            properties.isLow ? 1 : 0,
            properties.isZero ? 1 : 0,
            // NEW: Granular features for the AI
            properties.isD1 ? 1 : 0, // First dozen
            properties.isD2 ? 1 : 0, // Second dozen
            properties.isD3 ? 1 : 0, // Third dozen
            properties.isCol1 ? 1 : 0, // Column 1
            properties.isCol2 ? 1 : 0, // Column 2
            properties.isCol3 ? 1 : 0, // Column 3
            // End NEW features
            ...allPredictionTypes.map(type => item.typeSuccessStatus[type.id] ? 1 : 0)
        ];
        return features;
    };

    let rawFeatures = [];
    let rawGroupLabels = [];
    let rawFailureLabels = [];

    const featuresForScaling = validHistory.map(item => getFeatures(item));
    scaler = {
        min: Array(featuresForScaling[0].length).fill(Infinity),
        max: Array(featuresForScaling[0].length).fill(-Infinity)
    };
    featuresForScaling.forEach(row => {
        row.forEach((val, i) => {
            scaler.min[i] = Math.min(scaler.min[i], val);
            scaler.max[i] = Math.max(scaler.max[i], val);
        });
    });

    const scaleFeature = (value, index) => {
        const featureMin = scaler.min[index];
        const featureMax = scaler.max[index];
        if (featureMax === featureMin) return 0;
        return (value - featureMin) / (featureMax - featureMin);
    };

    for (let i = 0; i < validHistory.length - SEQUENCE_LENGTH; i++) {
        const sequence = validHistory.slice(i, i + SEQUENCE_LENGTH);
        const targetItem = validHistory[i + SEQUENCE_LENGTH];

        const xs_row = sequence.map(item => getFeatures(item).map((val, idx) => scaleFeature(val, idx)));
        rawFeatures.push(xs_row);

        // Label 1: Winning Groups
        const group_ys_row = allPredictionTypes.map(type => targetItem.typeSuccessStatus[type.id] ? 1 : 0);
        rawGroupLabels.push(group_ys_row);
        
        // Label 2: Failure Modes (one-hot encoded)
        const failure_ys_row = failureModes.map(mode => (targetItem.failureMode === mode ? 1 : 0));
        rawFailureLabels.push(failure_ys_row);
    }

    const featureCount = rawFeatures.length > 0 ? rawFeatures[0][0].length : 0;
    const groupLabelCount = allPredictionTypes.length;
    const failureLabelCount = failureModes.length;

    const xs = rawFeatures.length > 0 ? tf.tensor3d(rawFeatures) : null;
    // Create a dictionary of labels for the two outputs
    const ys = {
        group_output: rawGroupLabels.length > 0 ? tf.tensor2d(rawGroupLabels) : null,
        failure_output: rawFailureLabels.length > 0 ? tf.tensor2d(rawFailureLabels) : null
    };

    return { xs, ys, featureCount, groupLabelCount, failureLabelCount };
}

// Function to create the Multi-Output LSTM model architecture
function createMultiOutputLSTMModel(inputShape, groupOutputUnits, failureOutputUnits) {
    // Use the functional API to create a multi-output model
    const input = tf.input({shape: inputShape});

    // Shared LSTM layer
    const lstmLayer = tf.layers.lstm({
        units: LSTM_UNITS,
        returnSequences: false,
        activation: 'relu'
    }).apply(input);

    // Dropout layer for regularization
    const dropoutLayer = tf.layers.dropout({ rate: 0.2 }).apply(lstmLayer);

    // Head 1: Output for winning group prediction
    const groupOutput = tf.layers.dense({
        units: groupOutputUnits,
        activation: 'sigmoid',
        name: 'group_output' // Name must match the label key
    }).apply(dropoutLayer);

    // Head 2: Output for failure mode prediction
    const failureOutput = tf.layers.dense({
        units: failureOutputUnits,
        activation: 'softmax', // Softmax is used for multi-class classification
        name: 'failure_output' // Name must match the label key
    }).apply(dropoutLayer);

    // Create the model with one input and two outputs
    const model = tf.model({inputs: input, outputs: [groupOutput, failureOutput]});

    // Compile the model with separate loss functions for each output
    model.compile({
        optimizer: tf.train.adam(),
        loss: {
            'group_output': 'binaryCrossentropy',
            'failure_output': 'categoricalCrossentropy'
        },
        metrics: ['accuracy']
    });
    
    return model;
}

// Function to train the LSTM model
async function trainLSTMModel(historyData) {
    if (isTraining) {
        self.postMessage({ type: 'status', message: 'AI Model: Training already in progress.' });
        return;
    }
    isTraining = true;
    self.postMessage({ type: 'status', message: 'AI Model: Preparing data...' });

    const { xs, ys, featureCount, groupLabelCount, failureLabelCount } = prepareDataForLSTM(historyData);

    if (!xs || !ys.group_output || !ys.failure_output) {
        self.postMessage({ type: 'status', message: 'AI Model: Not enough valid data to train.' });
        if (mlModel) { mlModel.dispose(); mlModel = null; }
        await clearModelStorage();
        isTraining = false;
        return;
    }

    let modelToTrain;
    // Check compatibility for the new multi-output model
    const isModelCompatible = mlModel &&
                              mlModel.inputs[0].shape[1] === SEQUENCE_LENGTH &&
                              mlModel.inputs[0].shape[2] === featureCount &&
                              mlModel.outputs[0].shape[1] === groupLabelCount &&
                              mlModel.outputs[1].shape[1] === failureLabelCount;

    if (!isModelCompatible) {
        if (mlModel) {
            mlModel.dispose();
            console.log('Disposed incompatible TF.js model.');
        }
        modelToTrain = createMultiOutputLSTMModel([SEQUENCE_LENGTH, featureCount], groupLabelCount, failureLabelCount);
        console.log('TF.js Multi-Output Model created successfully in worker.');
    } else {
        console.log('Re-creating model and copying weights for continued training.');
        modelToTrain = createMultiOutputLSTMModel([SEQUENCE_LENGTH, featureCount], groupLabelCount, failureLabelCount);
        modelToTrain.setWeights(mlModel.getWeights());
        mlModel.dispose();
    }
    
    mlModel = modelToTrain;

    try {
        self.postMessage({ type: 'status', message: 'AI Model: Training...' });
        await mlModel.fit(xs, ys, {
            epochs: EPOCHS,
            batchSize: BATCH_SIZE,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    self.postMessage({ type: 'status', message: `AI Model: Training Epoch ${epoch + 1}/${EPOCHS} - Loss: ${logs.loss.toFixed(4)}` });
                }
            }
        });
        console.log('TF.js Model training complete in worker.');
        await saveModel(mlModel, scaler);
        self.postMessage({ type: 'status', message: 'AI Model: Ready!' });

    } catch (error) {
        console.error('Error during model training in worker:', error);
        self.postMessage({ type: 'status', message: `AI Model: Training failed! ${error.message}` });
        if (mlModel) { mlModel.dispose(); mlModel = null; }
        await clearModelStorage();
    } finally {
        xs.dispose();
        if (ys.group_output) ys.group_output.dispose();
        if (ys.failure_output) ys.failure_output.dispose();
        isTraining = false;
    }
}

// Function to predict using the multi-output model
async function predictGroupProbabilities(historyData) {
    if (!mlModel || !scaler) {
        return null;
    }
    const validHistory = historyData.filter(item => item.status === 'success' && item.winningNumber !== null);
    if (validHistory.length < SEQUENCE_LENGTH) {
        return null;
    }

    const lastSequence = validHistory.slice(-SEQUENCE_LENGTH);

    const getFeatures = (item) => {
        const properties = getNumberProperties(item.winningNumber);
        return [
            item.num1 / 36, item.num2 / 36, item.difference / 36,
            (item.num1 + item.num2) / 72,
            item.pocketDistance !== null ? item.pocketDistance / 18 : 0,
            // Normalized pocket distance for the *recommended* group
            item.recommendedGroupPocketDistance !== null ? item.recommendedGroupPocketDistance / 18 : 1,
            properties.isEven ? 1 : 0, properties.isOdd ? 1 : 0,
            properties.isRed ? 1 : 0, properties.isBlack ? 1 : 0,
            properties.isGreen ? 1 : 0, properties.isHigh ? 1 : 0,
            properties.isLow ? 1 : 0, properties.isZero ? 1 : 0,
            // Granular features for the AI
            properties.isD1 ? 1 : 0,
            properties.isD2 ? 1 : 0,
            properties.isD3 ? 1 : 0,
            properties.isCol1 ? 1 : 0,
            properties.isCol2 ? 1 : 0,
            properties.isCol3 ? 1 : 0,
        ];
    };

    const scaleFeature = (value, index) => {
        if (!scaler || scaler.min.length !== getFeatures(validHistory[0]).length) {
            console.error("Scaler is incompatible with new feature set. Retraining is required.");
            // Attempt to re-initialize scaler if incompatible (basic fallback)
            const newFeaturesSample = getFeatures(validHistory[0]);
            scaler = {
                min: Array(newFeaturesSample.length).fill(0), // Default to 0 if no min/max can be determined
                max: Array(newFeaturesSample.length).fill(1)  // Default to 1 for normalized features
            };
            self.postMessage({ type: 'status', message: 'AI Model: Scaler re-initialized. Please retrain.' });
            return value; // Return original value, will likely lead to poor prediction
        }
        const featureMin = scaler.min[index];
        const featureMax = scaler.max[index];
        if (featureMax === featureMin) return 0;
        return (value - featureMin) / (featureMax - featureMin);
    };

    const inputFeatures = lastSequence.map(item => getFeatures(item).map((val, idx) => scaleFeature(val, idx)));
    
    let inputTensor = null;
    let predictions = [];
    try {
        inputTensor = tf.tensor3d([inputFeatures]);
        predictions = mlModel.predict(inputTensor); // This will return an array of tensors
        
        const groupProbs = await predictions[0].data();
        const failureProbs = await predictions[1].data();

        const result = {
            groups: {},
            failures: {}
        };

        allPredictionTypes.forEach((type, index) => {
            result.groups[type.id] = groupProbs[index];
        });
        // Ensure failureModes is correctly defined and matches output units
        const definedFailureModes = ['none', 'normalLoss', 'streakBreak', 'sectionShift']; // Ensure consistency
        definedFailureModes.forEach((mode, index) => {
            result.failures[mode] = failureProbs[index];
        });
        
        return result;

    } catch (error) {
        console.error('Error during ML prediction in worker:', error);
        return null;
    } finally {
        if (inputTensor) inputTensor.dispose();
        if (predictions && Array.isArray(predictions)) {
            predictions.forEach(p => p.dispose());
        }
    }
}

// Save model and scaler
async function saveModel(modelToSave, currentScaler) {
    if (!modelToSave) return;
    try {
        self.postMessage({ type: 'saveScaler', payload: JSON.stringify(currentScaler) }); // Send scaler to main thread for local storage
        await modelToSave.save(`indexeddb://${TFJS_MODEL_STORAGE_KEY}`);
        console.log('TF.js model saved to IndexedDB from worker.');
    } catch (error) {
        console.error('Error saving TF.js model from worker:', error);
    }
}

// Load model from IndexedDB
async function loadModelFromStorage() {
    try {
        const loadedModel = await tf.loadLayersModel(`indexeddb://${TFJS_MODEL_STORAGE_KEY}/model.json`);
        console.log('TF.js model loaded from IndexedDB in worker.');
        return loadedModel;
    } catch (error) {
        console.warn('No TF.js model found in IndexedDB or error loading in worker.');
        return null;
    }
}

// Clear model from IndexedDB
async function clearModelStorage() {
    try {
        await tf.io.removeModel(`indexeddb://${TFJS_MODEL_STORAGE_KEY}`);
        scaler = null; // Clear scaler as well
        console.log('TF.js model and scaler cleared from storage by worker.');
    } catch (error) {
        console.error('Error clearing TF.js model storage from worker:', error);
    }
}

// --- Message Handling for Web Worker ---
self.onmessage = async (event) => {
    const { type, payload } = event.data;

    switch (type) {
        case 'init':
            allPredictionTypes = payload.allPredictionTypes;
            terminalMapping = payload.terminalMapping;
            rouletteWheel = payload.rouletteWheel;
            scaler = payload.scaler ? JSON.parse(payload.scaler) : null; // Load scaler from main thread
            mlModel = await loadModelFromStorage();
            if (mlModel) {
                self.postMessage({ type: 'status', message: 'AI Model: Ready!' });
            } else {
                self.postMessage({ type: 'status', message: `AI Model: Need at least ${TRAINING_MIN_HISTORY} confirmed spins to train.` });
            }
            break;

        case 'train':
            await trainLSTMModel(payload.history);
            break;

        case 'predict':
            const probabilities = await predictGroupProbabilities(payload.history);
            self.postMessage({ type: 'predictionResult', probabilities });
            break;

        case 'clear_model':
            if (mlModel) { mlModel.dispose(); mlModel = null; }
            await clearModelStorage();
            self.postMessage({ type: 'status', message: 'AI Model: Cleared.' });
            break;

        case 'update_config':
            // Update local config (used for getFeatures in predict and train)
            allPredictionTypes = payload.allPredictionTypes;
            // No longer passing currentNum1, etc., as worker doesn't need it for local AI
            break;
    }
};
