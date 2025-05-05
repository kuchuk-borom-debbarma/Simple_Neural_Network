namespace Simple_neural_network;

public class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Simple Neural Network Demo");
        Console.WriteLine("=========================\n");

        // Generate the dataset
        Console.WriteLine("Generating dataset...");
        var data = new SimpleDatasetGenerator().Generate(1000000);
        Console.WriteLine($"Generated {data.Count} data points\n");

        // Display a few sample points
        Console.WriteLine("Sample data points (X, Y => Z):");
        for (int i = 0; i < 3; i++)
        {
            Console.WriteLine(
                $"  {data[i].X:F2}, {data[i].Y:F2} => {data[i].Z:F2} (expected: {data[i].X + 2 * data[i].Y:F2})");
        }

        Console.WriteLine();

        // Split the data into training and test sets
        var (trainingData, testData) = DataProcessor.SplitData(data, 0.2);
        Console.WriteLine($"Split data into {trainingData.Count} training and {testData.Count} test samples\n");

        // Normalize the data
        Console.WriteLine("Normalizing data...");
        var scaler = new DataProcessor.MinMaxScaler();
        scaler.Fit(data);

        // Prepare data for training and testing
        var (trainingInputs, trainingOutputs) = DataProcessor.PrepareData(trainingData, scaler);
        var (testInputs, testOutputs) = DataProcessor.PrepareData(testData, scaler);

        // Create a configurable neural network
        // Here we use a network with 2 inputs (X,Y), 1 hidden layer with 10 neurons, and 1 output (Z)
        Console.WriteLine("Creating neural network architecture...");
        var network = new NeuralNetwork(new[] { 2, 10, 1 })
        {
            LearningRate = 0.01,
            MaxEpochs = 5000,
            ConvergenceThreshold = 0.0000001
        };

        // Train the network
        Console.WriteLine("\nTraining neural network...");
        var history = network.Train(trainingInputs, trainingOutputs, batchSize: 32, validationSplit: 0.1);
        history.PrintSummary();

        // Evaluate the network
        Console.WriteLine("\nEvaluating on test data...");
        var evaluation = DataProcessor.EvaluateModel(network, testInputs, testOutputs, scaler);
        evaluation.PrintSummary();

        // Try different network architectures
        Console.WriteLine("\n\nComparing Different Network Architectures");
        Console.WriteLine("======================================\n");

        CompareNetworkArchitectures(
            trainingInputs, trainingOutputs,
            testInputs, testOutputs,
            scaler,
            new[]
            {
                new[] { 2, 5, 1 },
                new[] { 2, 10, 1 },
                new[] { 2, 5, 5, 1 },
                new[] { 2, 15, 10, 1 }
            }
        );

        // Try to predict some custom inputs
        Console.WriteLine("\n\nPredicting Custom Inputs");
        Console.WriteLine("=======================\n");

        var customInputs = new[]
        {
            new double[] { 5, 3 },
            new double[] { -2, 4 },
            new double[] { 0, 0 },
            new double[] { 8, -6 }
        };

        foreach (var input in customInputs)
        {
            // Scale the input
            var scaledInput = new[]
            {
                scaler.Scale("X", input[0]),
                scaler.Scale("Y", input[1])
            };

            // Generate prediction
            var output = network.Forward(scaledInput);

            // Unscale the output
            var prediction = scaler.Unscale("Z", output[0]);
            var expected = input[0] + 2 * input[1];

            Console.WriteLine($"Input: [{input[0]}, {input[1]}]");
            Console.WriteLine($"  Predicted Z: {prediction:F4}");
            Console.WriteLine($"  Expected Z:  {expected:F4}");
            Console.WriteLine($"  Error:       {Math.Abs(prediction - expected):F4}\n");
        }

        Console.WriteLine("Demo completed. Press any key to exit.");
        Console.ReadKey();
    }

    private static void CompareNetworkArchitectures(
        List<double[]> trainingInputs,
        List<double[]> trainingOutputs,
        List<double[]> testInputs,
        List<double[]> testOutputs,
        DataProcessor.MinMaxScaler scaler,
        int[][] architectures)
    {
        foreach (var architecture in architectures)
        {
            // Create a neural network with the specified architecture
            string archDescription = string.Join("-", architecture);
            Console.WriteLine($"Testing architecture: {archDescription}");

            var network = new NeuralNetwork(architecture)
            {
                LearningRate = 0.01,
                MaxEpochs = 300,
                ConvergenceThreshold = 0.000001
            };

            // Train the network
            var startTime = DateTime.Now;
            var history = network.Train(trainingInputs, trainingOutputs, batchSize: 32, validationSplit: 0.1);
            var trainingTime = (DateTime.Now - startTime).TotalSeconds;

            // Evaluate the network
            var evaluation = DataProcessor.EvaluateModel(network, testInputs, testOutputs, scaler);

            Console.WriteLine($"  Architecture {archDescription}:");
            Console.WriteLine($"    Training time: {trainingTime:F2} seconds");
            Console.WriteLine($"    RMSE: {evaluation.RootMeanSquaredError:F6}");
            Console.WriteLine($"    MAE: {evaluation.MeanAbsoluteError:F6}\n");
        }
    }
}