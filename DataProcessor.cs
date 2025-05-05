
using Simple_neural_network.model;

namespace Simple_neural_network;

public class DataProcessor
{
    /// <summary>
    /// Normalizes the data to values between 0 and 1
    /// </summary>
    public class MinMaxScaler
    {
        private readonly Dictionary<string, double> _mins = new Dictionary<string, double>();
        private readonly Dictionary<string, double> _maxs = new Dictionary<string, double>();
            
        public void Fit(List<DataPoint> dataPoints)
        {
            _mins["X"] = dataPoints.Min(d => d.X);
            _maxs["X"] = dataPoints.Max(d => d.X);
                
            _mins["Y"] = dataPoints.Min(d => d.Y);
            _maxs["Y"] = dataPoints.Max(d => d.Y);
                
            _mins["Z"] = dataPoints.Min(d => d.Z);
            _maxs["Z"] = dataPoints.Max(d => d.Z);
        }
            
        public double Scale(string feature, double value)
        {
            if (!_mins.ContainsKey(feature) || !_maxs.ContainsKey(feature))
                throw new ArgumentException($"Feature {feature} not found in scaler");
                
            double min = _mins[feature];
            double max = _maxs[feature];
                
            // Avoid division by zero
            if (Math.Abs(max - min) < 1e-10)
                return 0.5;
                
            return (value - min) / (max - min);
        }
            
        public double Unscale(string feature, double value)
        {
            if (!_mins.ContainsKey(feature) || !_maxs.ContainsKey(feature))
                throw new ArgumentException($"Feature {feature} not found in scaler");
                
            double min = _mins[feature];
            double max = _maxs[feature];
                
            return (value * (max - min)) + min;
        }
    }
        
    /// <summary>
    /// Prepares data for the neural network by splitting it into training and test sets
    /// </summary>
    public static (List<DataPoint> TrainingData, List<DataPoint> TestData) SplitData(List<DataPoint> data, double testRatio = 0.2)
    {
        Random random = new Random(42); // Using a fixed seed for reproducibility
        List<DataPoint> shuffledData = data.OrderBy(_ => random.Next()).ToList();
            
        int testSize = (int)(shuffledData.Count * testRatio);
        int trainingSize = shuffledData.Count - testSize;
            
        return (
            shuffledData.GetRange(0, trainingSize),
            shuffledData.GetRange(trainingSize, testSize)
        );
    }
        
    /// <summary>
    /// Prepares data for the neural network by converting DataPoints to input and output arrays
    /// </summary>
    public static (List<double[]> Inputs, List<double[]> Outputs) PrepareData(
        List<DataPoint> dataPoints, 
        MinMaxScaler scaler = null)
    {
        var inputs = new List<double[]>();
        var outputs = new List<double[]>();
            
        foreach (var dataPoint in dataPoints)
        {
            double[] input;
            double[] output;
                
            if (scaler != null)
            {
                input = new[]
                {
                    scaler.Scale("X", dataPoint.X),
                    scaler.Scale("Y", dataPoint.Y)
                };
                    
                output = new[]
                {
                    scaler.Scale("Z", dataPoint.Z)
                };
            }
            else
            {
                input = new[] { dataPoint.X, dataPoint.Y };
                output = new[] { dataPoint.Z };
            }
                
            inputs.Add(input);
            outputs.Add(output);
        }
            
        return (inputs, outputs);
    }
        
    /// <summary>
    /// Evaluates the model on test data and returns performance metrics
    /// </summary>
    public static ModelEvaluation EvaluateModel(
        NeuralNetwork model,
        List<double[]> inputs,
        List<double[]> targets,
        MinMaxScaler scaler = null)
    {
        double totalMSE = 0;
        double totalMAE = 0;
        List<double> predictions = new List<double>();
        List<double> actuals = new List<double>();
            
        for (int i = 0; i < inputs.Count; i++)
        {
            double[] output = model.Forward(inputs[i]);
            double predicted = output[0];
            double actual = targets[i][0];
                
            // Unscale values if a scaler was provided
            if (scaler != null)
            {
                predicted = scaler.Unscale("Z", predicted);
                actual = scaler.Unscale("Z", actual);
            }
                
            predictions.Add(predicted);
            actuals.Add(actual);
                
            double error = predicted - actual;
            totalMSE += error * error;
            totalMAE += Math.Abs(error);
        }
            
        double mse = totalMSE / inputs.Count;
        double mae = totalMAE / inputs.Count;
        double rmse = Math.Sqrt(mse);
            
        return new ModelEvaluation
        {
            MeanSquaredError = mse,
            MeanAbsoluteError = mae,
            RootMeanSquaredError = rmse,
            Predictions = predictions,
            ActualValues = actuals
        };
    }
}
    
public class ModelEvaluation
{
    public double MeanSquaredError { get; set; }
    public double MeanAbsoluteError { get; set; }
    public double RootMeanSquaredError { get; set; }
    public List<double> Predictions { get; set; }
    public List<double> ActualValues { get; set; }
        
    public void PrintSummary()
    {
        Console.WriteLine($"Model Evaluation Metrics:");
        Console.WriteLine($"  Mean Squared Error (MSE): {MeanSquaredError:F6}");
        Console.WriteLine($"  Root Mean Squared Error (RMSE): {RootMeanSquaredError:F6}");
        Console.WriteLine($"  Mean Absolute Error (MAE): {MeanAbsoluteError:F6}");
            
        // Print some sample predictions
        int samplesToShow = Math.Min(5, Predictions.Count);
        Console.WriteLine("\nSample Predictions vs Actual Values:");
        Console.WriteLine("  Predicted\t\tActual\t\tError");
            
        for (int i = 0; i < samplesToShow; i++)
        {
            double error = Predictions[i] - ActualValues[i];
            Console.WriteLine($"  {Predictions[i]:F4}\t\t{ActualValues[i]:F4}\t\t{error:F4}");
        }
    }
}