//
//  ViewController.swift
//  ActivityClassifier
//
//  Created by Meghan Kane on 7/4/18.
//  Copyright © 2018 meghaphone. All rights reserved.
//

import UIKit
import CoreMotion
import CoreML

struct ModelConstants {
    static let numOfFeatures = 6
    static let predictionWindowSize = 50
    static let sensorsUpdateInterval = 1.0 / 50.0
    static let hiddenInLength = 200
    static let hiddenCellInLength = 200
}

class ViewController: UIViewController {
    
    @IBOutlet var activityLabel: UILabel!
    @IBOutlet var probabilityLabel: UILabel!
    
    let activityClassificationModel = ActivityClassifier()
    let motionManager = CMMotionManager()
    var currentIndexInPredictionWindow = 0
    let predictionWindowDataArray = try? MLMultiArray(shape: [1, ModelConstants.predictionWindowSize, ModelConstants.numOfFeatures] as [NSNumber], dataType: MLMultiArrayDataType.double)
    var lastHiddenOutput = try? MLMultiArray(shape:[ModelConstants.hiddenInLength as NSNumber], dataType: MLMultiArrayDataType.double)
    var lastHiddenCellOutput = try? MLMultiArray(shape:[ModelConstants.hiddenCellInLength as NSNumber], dataType: MLMultiArrayDataType.double)
    var currentIndexUpdated: [Bool] = [false, false, false, false, false, false]

    // MARK - IBActions
    
    @IBAction func startButtonPressed() {
        startProcessingSensorData()
    }
    
    @IBAction func stopButtonPressed() {
        stopProcessingSensorData()
        setLabelsToDefaultText()
    }
    
    // MARK - Process CoreMotion sensor data
    
    func startProcessingSensorData() {
        guard motionManager.isAccelerometerAvailable && motionManager.isGyroAvailable else { return }
        
        motionManager.accelerometerUpdateInterval = TimeInterval(ModelConstants.sensorsUpdateInterval)
        motionManager.gyroUpdateInterval = TimeInterval(ModelConstants.sensorsUpdateInterval)
        
        motionManager.startAccelerometerUpdates(to: .main) { accelerometerData, error in
            guard let accelerometerData = accelerometerData else { return }

            self.addAccelSampleToDataArray(accelSample: accelerometerData)
        }
        
        motionManager.startGyroUpdates(to: .main) { gyroData, error in
            guard let gyroData = gyroData else { return }
            
            self.addGyroSampleToDataArray(gyroSample: gyroData)
        }
    }
    
    func stopProcessingSensorData() {
        motionManager.stopAccelerometerUpdates()
        motionManager.stopGyroUpdates()
    }
    
    func addAccelSampleToDataArray(accelSample: CMAccelerometerData) {
        guard let dataArray = predictionWindowDataArray else { return }
        
        addDataPointToDataArray(featureNumber: 0, dataSampleValue: NSNumber(value: accelSample.acceleration.x), dataArray: dataArray)
        addDataPointToDataArray(featureNumber: 1, dataSampleValue: NSNumber(value: accelSample.acceleration.y), dataArray: dataArray)
        addDataPointToDataArray(featureNumber: 2, dataSampleValue: NSNumber(value: accelSample.acceleration.z), dataArray: dataArray)
        
        // Update the index in the prediction window data array
        updatePredictionWindowIfNecessary(index: currentIndexInPredictionWindow)
    }
    
    func addGyroSampleToDataArray(gyroSample: CMGyroData) {
        // Add the current accelerometer reading to the data array
        guard let dataArray = predictionWindowDataArray else { return }
        
        addDataPointToDataArray(featureNumber: 3, dataSampleValue: NSNumber(value: gyroSample.rotationRate.x), dataArray: dataArray)
        addDataPointToDataArray(featureNumber: 4, dataSampleValue: NSNumber(value: gyroSample.rotationRate.y), dataArray: dataArray)
        addDataPointToDataArray(featureNumber: 5, dataSampleValue: NSNumber(value: gyroSample.rotationRate.z), dataArray: dataArray)
    
        // Update the index in the prediction window data array
        updatePredictionWindowIfNecessary(index: currentIndexInPredictionWindow)
    }
    
    func updatePredictionWindowIfNecessary(index: Int) {
        guard predictionWindowDataArray != nil else { return }
        
        for index in currentIndexUpdated {
            if index == false {
                return
            }
        }
        
        currentIndexUpdated = [false, false, false, false, false, false]
        
        // Update the index in the prediction window data array
        currentIndexInPredictionWindow += 1
        
        if (currentIndexInPredictionWindow == ModelConstants.predictionWindowSize) {
            performModelPrediction()
            
            // Start a new prediction window
            currentIndexInPredictionWindow = 0
        }
    }
    
    func performModelPrediction() {
        guard let dataArray = predictionWindowDataArray else { return }
        
        // 1. Perform model prediction
        let modelPrediction = try? activityClassificationModel.prediction(features: dataArray,
                                                                          hiddenIn: lastHiddenOutput,
                                                                          cellIn: lastHiddenCellOutput)
        
        // 2. Update the state vectors
        lastHiddenOutput = modelPrediction?.hiddenOut
        lastHiddenCellOutput = modelPrediction?.cellOut
        
        // 3. Return the predicted activity - the activity with the highest probability
        guard let topActivity = modelPrediction?.activity else {
            setLabelsToDefaultText()
            return
        }
        
        // 4. Update UI
        activityLabel.text = topActivity
        probabilityLabel.text = "\(String(describing: modelPrediction?.activityProbability[topActivity]))"
    }
    
    private func setLabelsToDefaultText() {
        activityLabel.text = "🤷🏼‍♀️"
        probabilityLabel.text = "🤷🏼‍♀️"
    }
}

private extension ViewController {
    func addDataPointToDataArray(featureNumber: Int, dataSampleValue:NSNumber, dataArray: MLMultiArray) {
        let dataPoint: [NSNumber] = [NSNumber(value: 0), NSNumber(value: currentIndexInPredictionWindow), NSNumber(value: featureNumber)]
        dataArray[dataPoint] = dataSampleValue
        currentIndexUpdated[featureNumber] = true
    }
}
