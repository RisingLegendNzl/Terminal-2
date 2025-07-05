// aiWorker.js - Web Worker for TensorFlow.js AI Model (Ensemble)

importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs');

// --- ENSEMBLE CONFIGURATION ---
const ENSEMBLE_CONFIG = [
    {
        name: 'Specialist',
        path: 'roulette-ml-model-specialist',
        lstmUnits: 16, // Smaller, faster model
        epochs: 40,
        batchSize: 32,
    },
    {
        name: 'Generalist',
        path: 'roulette-ml-model-generalist',
        lstmUnits: 64, // Larger, more complex model
        epochs: 60,
        batchSize: 16,
    }
];

const SEQUENCE_LENGTH = 5;
const TRAINING_MIN_HISTORY = 10;
const failureModes = ['none', 'normalLoss', 'streakBreak', 'sectionShift'];

let ensemble = ENSEMBLE_CONFIG.map(config => ({ ...config, model: null, scaler: null }));
let allPredictionTypes = [];
let isTraining = false;

// Helper to get number properties (unchanged)
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
    const isD1 = num >= 1 && num <= 12;
    const isD2 = num >= 13 && num <= 24;
    const isD3 = num >= 25 && num <= 36;
    const isCol1 = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34].includes(num);
    const isCol2 = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35].includes(num);
    const isCol3 = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36].includes(num);
    return {
        isEven: isEven ? 1 : 0, isOdd: isOdd ? 1 : 0, isRed: color === 'red' ? 1 : 0,
        isBlack: color === 'black' ? 1 : 0, isHigh: isHigh ? 1 : 0, isLow: isLow ? 1 : 0,
        isD1: isD1 ? 1 : 0, isD2: isD2 ? 1 : 0, isD3: isD3 ? 1 : 0,
        isCol1: isCol1 ? 1 : 0, isCol2: isCol2 ? 1 : 0, isCol3: isCol3 ? 1 : 0,
    };
}

// Function to prepare data (largely unchanged, creates a universal scaler)
function prepareDataForLSTM(historyData) {
    const validHistory = historyData.filter(item => item.status === 'success' && item.winningNumber !== null);
    if (validHistory.length < SEQUENCE_LENGTH + 1) {
        self.postMessage({ type: 'status', message: `AI Model: Need at least ${TRAINING_MIN_HISTORY} confirmed spins to train.` });
        return { xs: null, ys: null, scaler: null, featureCount: 0 };
    }

    const getFeatures = (item) => {
        const props = getNumberProperties(item.winningNumber);
        return [
            item.num1 / 36, item.num2 / 36, item.difference / 36,
            item.pocketDistance !== null ? item.pocketDistance / 18 : 0,
            item.recommendedGroupPocketDistance !== null ? item.recommendedGroupPocketDistance / 18 : 1,
            ...Object.values(props),
            ...allPredictionTypes.map(type => item.typeSuccessStatus[type.id] ? 1 : 0)
        ];
    };
    
    const featuresForScaling = validHistory.map(item => getFeatures(item));
    const newScaler = {
        min: Array(featuresForScaling[0].length).fill(Infinity),
        max: Array(featuresForScaling[0].length).fill(-Infinity)
    };
    featuresForScaling.forEach(row => {
        row.forEach((val, i) => {
            newScaler.min[i] = Math.min(newScaler.min[i], val);
            newScaler.max[i] = Math.max(newScaler.max[i], val);
        });
    });

    const scaleFeature = (value, index) => {
        const featureMin = newScaler.min[index];
        const featureMax = newScaler.max[index];
        if (featureMax === featureMin) return 0;
        return (value - featureMin) / (featureMax - featureMin);
    };

    let rawFeatures = [];
    let rawGroupLabels = [];
    let rawFailureLabels = [];

    for (let i = 0; i < validHistory.length - SEQUENCE_LENGTH; i++) {
        const sequence = validHistory.slice(i, i + SEQUENCE_LENGTH);
        const targetItem = validHistory[i + SEQUENCE_LENGTH];
        const xs_row = sequence.map(item => getFeatures(item).map((val, idx) => scaleFeature(val, idx)));
        rawFeatures.push(xs_row);
        rawGroupLabels.push(allPredictionTypes.map(type => targetItem.typeSuccessStatus[type.id] ? 1 : 0));
        rawFailureLabels.push(failureModes.map(mode => (targetItem.failureMode === mode ? 1 : 0)));
    }

    const featureCount = rawFeatures.length > 0 ? rawFeatures[0][0].length : 0;
    const xs = rawFeatures.length > 0 ? tf.tensor3d(rawFeatures) : null;
    const ys = {
        group_output: rawGroupLabels.length > 0 ? tf.tensor2d(rawGroupLabels) : null,
        failure_output: rawFailureLabels.length > 0 ? tf.tensor2d(rawFailureLabels) : null
    };

    return { xs, ys, scaler: newScaler, featureCount };
}

// Function to create model (now accepts lstmUnits as a parameter)
function createMultiOutputLSTMModel(inputShape, groupOutputUnits, failureOutputUnits, lstmUnits) {
    const input = tf.input({ shape: inputShape });
    const lstmLayer = tf.layers.lstm({ units: lstmUnits, returnSequences: false, activation: 'relu' }).apply(input);
    const dropoutLayer = tf.layers.dropout({ rate: 0.2 }).apply(lstmLayer);
    const groupOutput = tf.layers.dense({ units: groupOutputUnits, activation: 'sigmoid', name: 'group_output' }).apply(dropoutLayer);
    const failureOutput = tf.layers.dense({ units: failureOutputUnits, activation: 'softmax', name: 'failure_output' }).apply(dropoutLayer);
    const model = tf.model({ inputs: input, outputs: [groupOutput, failureOutput] });
    model.compile({
        optimizer: tf.train.adam(),
        loss: { 'group_output': 'binaryCrossentropy', 'failure_output': 'categoricalCrossentropy' },
        metrics: ['accuracy']
    });
    return model;
}

// Main training function (overhauled for ensemble)
async function trainEnsemble(historyData) {
    if (isTraining) {
        self.postMessage({ type: 'status', message: 'AI Ensemble: Training already in progress.' });
        return;
    }
    isTraining = true;
    self.postMessage({ type: 'status', message: 'AI Ensemble: Preparing data...' });

    const { xs, ys, scaler, featureCount } = prepareDataForLSTM(historyData);
    if (!xs) {
        self.postMessage({ type: 'status', message: 'AI Ensemble: Not enough valid data to train.' });
        isTraining = false;
        return;
    }

    // A single scaler is now used for all models
    self.postMessage({ type: 'saveScaler', payload: JSON.stringify(scaler) });
    ensemble.forEach(member => member.scaler = scaler);

    const groupLabelCount = allPredictionTypes.length;
    const failureLabelCount = failureModes.length;

    for (const member of ensemble) {
        try {
            self.postMessage({ type: 'status', message: `AI Ensemble: Training ${member.name}...` });
            if (member.model) member.model.dispose(); // Dispose old model before training
            
            member.model = createMultiOutputLSTMModel([SEQUENCE_LENGTH, featureCount], groupLabelCount, failureLabelCount, member.lstmUnits);
            
            await member.model.fit(xs, ys, {
                epochs: member.epochs,
                batchSize: member.batchSize,
                callbacks: {
                    onEpochEnd: (epoch) => {
                        self.postMessage({ type: 'status', message: `AI Ensemble: Training ${member.name} (Epoch ${epoch + 1}/${member.epochs})` });
                    }
                }
            });
            await member.model.save(`indexeddb://${member.path}`);
            console.log(`TF.js Model ${member.name} saved.`);
        } catch (error) {
            console.error(`Error training model ${member.name}:`, error);
            self.postMessage({ type: 'status', message: `AI Ensemble: Training for ${member.name} failed.` });
        }
    }

    xs.dispose();
    if (ys.group_output) ys.group_output.dispose();
    if (ys.failure_output) ys.failure_output.dispose();
    isTraining = false;
    self.postMessage({ type: 'status', message: 'AI Ensemble: Ready!' });
}

// Prediction function (overhauled for ensemble)
async function predictWithEnsemble(historyData) {
    const activeModels = ensemble.filter(m => m.model && m.scaler);
    if (activeModels.length === 0) return null;

    const validHistory = historyData.filter(item => item.status === 'success' && item.winningNumber !== null);
    if (validHistory.length < SEQUENCE_LENGTH) return null;

    const lastSequence = validHistory.slice(-SEQUENCE_LENGTH);
    
    // Use the scaler from the first available model (they are all the same now)
    const scaler = activeModels[0].scaler;
    
    const getFeatures = (item) => {
        const props = getNumberProperties(item.winningNumber);
         return [
            item.num1 / 36, item.num2 / 36, item.difference / 36,
            item.pocketDistance !== null ? item.pocketDistance / 18 : 0,
            item.recommendedGroupPocketDistance !== null ? item.recommendedGroupPocketDistance / 18 : 1,
            ...Object.values(props),
            ...allPredictionTypes.map(type => item.typeSuccessStatus[type.id] ? 1 : 0)
        ];
    };
    
    const scaleFeature = (value, index) => {
        if (!scaler) return value;
        const featureMin = scaler.min[index];
        const featureMax = scaler.max[index];
        if (featureMax === featureMin) return 0;
        return (value - featureMin) / (featureMax - featureMin);
    };

    let inputTensor = null;
    try {
        const inputFeatures = lastSequence.map(item => getFeatures(item).map((val, idx) => scaleFeature(val, idx)));
        inputTensor = tf.tensor3d([inputFeatures]);

        const allPredictions = await Promise.all(activeModels.map(m => m.model.predict(inputTensor)));

        // Average the predictions
        const averagedGroupProbs = new Float32Array(allPredictionTypes.length).fill(0);
        const averagedFailureProbs = new Float32Array(failureModes.length).fill(0);

        for (const prediction of allPredictions) {
            const groupProbs = await prediction[0].data();
            const failureProbs = await prediction[1].data();
            groupProbs.forEach((p, i) => averagedGroupProbs[i] += p);
            failureProbs.forEach((p, i) => averagedFailureProbs[i] += p);
            prediction[0].dispose();
            prediction[1].dispose();
        }

        averagedGroupProbs.forEach((p, i) => averagedGroupProbs[i] /= allPredictions.length);
        averagedFailureProbs.forEach((p, i) => averagedFailureProbs[i] /= allPredictions.length);

        const finalResult = { groups: {}, failures: {} };
        allPredictionTypes.forEach((type, i) => finalResult.groups[type.id] = averagedGroupProbs[i]);
        failureModes.forEach((mode, i) => finalResult.failures[mode] = averagedFailureProbs[i]);
        
        return finalResult;

    } catch (error) {
        console.error('Error during ensemble prediction:', error);
        return null;
    } finally {
        if (inputTensor) inputTensor.dispose();
    }
}


// Storage functions (updated for ensemble)
async function loadModelsFromStorage() {
    const loadPromises = ensemble.map(async (member) => {
        try {
            member.model = await tf.loadLayersModel(`indexeddb://${member.path}/model.json`);
            console.log(`TF.js Model ${member.name} loaded from IndexedDB.`);
            return true;
        } catch (error) {
            console.warn(`Could not load model ${member.name}. It may need to be trained.`);
            return false;
        }
    });
    return Promise.all(loadPromises);
}

async function clearModelsFromStorage() {
    const clearPromises = ensemble.map(async (member) => {
        try {
            if (member.model) {
                member.model.dispose();
                member.model = null;
            }
            await tf.io.removeModel(`indexeddb://${member.path}`);
        } catch (error) {
            // Error is expected if model doesn't exist, so we don't log it as a critical failure
        }
    });
    await Promise.all(clearPromises);
    ensemble.forEach(m => m.scaler = null);
    console.log('All TF.js models and scalers cleared.');
}


// --- Message Handling for Web Worker ---
self.onmessage = async (event) => {
    const { type, payload } = event.data;
    switch (type) {
        case 'init':
            allPredictionTypes = payload.allPredictionTypes;
            const loadedScaler = payload.scaler ? JSON.parse(payload.scaler) : null;
            if(loadedScaler) {
                ensemble.forEach(m => m.scaler = loadedScaler);
            }
            const loadResults = await loadModelsFromStorage();
            if (loadResults.every(Boolean)) { // Only ready if ALL models loaded
                self.postMessage({ type: 'status', message: 'AI Ensemble: Ready!' });
            } else {
                self.postMessage({ type: 'status', message: `AI Ensemble: Need at least ${TRAINING_MIN_HISTORY} confirmed spins to train.` });
            }
            break;
        case 'train':
            await trainEnsemble(payload.history);
            break;
        case 'predict':
            const probabilities = await predictWithEnsemble(payload.history);
            self.postMessage({ type: 'predictionResult', probabilities });
            break;
        case 'clear_model':
            await clearModelsFromStorage();
            self.postMessage({ type: 'status', message: 'AI Ensemble: Cleared.' });
            break;
        case 'update_config':
            allPredictionTypes = payload.allPredictionTypes;
            break;
    }
};
