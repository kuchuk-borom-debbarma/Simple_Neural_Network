namespace Simple_neural_network;

public class NeuralNetwork
{
    private readonly int[] _layerSizes;
    private readonly double[][] _neurons;
    private readonly double[][][] _weights;
    private readonly double[][] _biases;
    private readonly Random _random = new Random();

    // Learning parameters
    public double LearningRate { get; set; } = 0.01;
    public int MaxEpochs { get; set; } = 1000;
    public double ConvergenceThreshold { get; set; } = 0.0001;

    /// <summary>
    /// Constructor for a configurable neural network
    /// </summary>
    /// <param name="layerSizes">Array containing the number of neurons in each layer (including input and output layers)</param>
    public NeuralNetwork(int[] layerSizes)
    {
        if (layerSizes.Length < 2)
            throw new ArgumentException("Neural network must have at least an input and output layer");

        _layerSizes = layerSizes;

        // Initialize neurons for each layer
        _neurons = new double[layerSizes.Length][];
        for (int i = 0; i < layerSizes.Length; i++)
        {
            _neurons[i] = new double[layerSizes[i]];
        }

        // Initialize weights and biases
        _weights = new double[layerSizes.Length - 1][][];
        _biases = new double[layerSizes.Length - 1][];

        // For each layer (except the output layer)
        for (int i = 0; i < layerSizes.Length - 1; i++)
        {
            _weights[i] = new double[layerSizes[i]][];
            _biases[i] = new double[layerSizes[i + 1]];

            // Initialize with random values using Xavier initialization
            double weightScale = Math.Sqrt(2.0 / (layerSizes[i] + layerSizes[i + 1]));

            // For each neuron in the current layer
            for (int j = 0; j < layerSizes[i]; j++)
            {
                _weights[i][j] = new double[layerSizes[i + 1]];

                // For each neuron in the next layer
                for (int k = 0; k < layerSizes[i + 1]; k++)
                {
                    _weights[i][j][k] = (RandomDouble() * 2 - 1) * weightScale;
                }
            }

            // Initialize biases with small random values
            for (int j = 0; j < layerSizes[i + 1]; j++)
            {
                _biases[i][j] = (RandomDouble() * 2 - 1) * 0.1;
            }
        }
    }

    /// <summary>
    /// Forward pass through the neural network
    /// </summary>
    /// <param name="inputs">Input values</param>
    /// <returns>Output values</returns>
    public double[] Forward(double[] inputs)
    {
        if (inputs.Length != _layerSizes[0])
            throw new ArgumentException($"Expected {_layerSizes[0]} inputs, but got {inputs.Length}");

        // Set input layer
        Array.Copy(inputs, _neurons[0], inputs.Length);

        // Forward pass through each layer
        for (int i = 0; i < _layerSizes.Length - 1; i++)
        {
            for (int j = 0; j < _layerSizes[i + 1]; j++)
            {
                _neurons[i + 1][j] = _biases[i][j];
                for (int k = 0; k < _layerSizes[i]; k++)
                {
                    _neurons[i + 1][j] += _neurons[i][k] * _weights[i][k][j];
                }

                // Apply activation function (ReLU for hidden layers, linear for output)
                if (i < _layerSizes.Length - 2)
                {
                    _neurons[i + 1][j] = ReLU(_neurons[i + 1][j]);
                }
            }
        }

        // Return output layer
        return _neurons[_neurons.Length - 1];
    }

    /// <summary>
    /// Train the neural network using backpropagation
    /// </summary>
    /// <param name="inputs">Input data as a list of arrays</param>
    /// <param name="targets">Target output data as a list of arrays</param>
    /// <param name="batchSize">Size of mini-batches</param>
    /// <param name="validationSplit">Fraction of data to use for validation (0.0-1.0)</param>
    /// <returns>Training history with loss values</returns>
    public TrainingHistory Train(List<double[]> inputs, List<double[]> targets, int batchSize = 32,
        double validationSplit = 0.2)
    {
        if (inputs.Count != targets.Count)
            throw new ArgumentException("Number of inputs must match number of targets");

        if (inputs.Count == 0)
            throw new ArgumentException("Training data cannot be empty");

        if (targets[0].Length != _layerSizes[_layerSizes.Length - 1])
            throw new ArgumentException(
                $"Expected {_layerSizes[_layerSizes.Length - 1]} output values, but got {targets[0].Length}");

        // Shuffle and split data
        var indices = Enumerable.Range(0, inputs.Count).ToList();
        Shuffle(indices);

        int validationCount = (int)(inputs.Count * validationSplit);
        int trainingCount = inputs.Count - validationCount;

        var trainingIndices = indices.GetRange(0, trainingCount);
        var validationIndices = indices.GetRange(trainingCount, validationCount);

        var history = new TrainingHistory();
        double previousLoss = double.MaxValue;

        // Training loop
        for (int epoch = 0; epoch < MaxEpochs; epoch++)
        {
            Shuffle(trainingIndices);

            // Mini-batch training
            for (int b = 0; b < trainingCount; b += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, trainingCount - b);

                // Process mini-batch
                for (int i = 0; i < actualBatchSize; i++)
                {
                    int idx = trainingIndices[b + i];
                    Backpropagate(inputs[idx], targets[idx]);
                }
            }

            // Calculate loss on training set
            double trainingLoss = CalculateLoss(inputs, targets, trainingIndices);

            // Calculate loss on validation set if we have one
            double validationLoss = validationCount > 0 ? CalculateLoss(inputs, targets, validationIndices) : 0;

            history.AddEpoch(epoch, trainingLoss, validationLoss);

            // Early stopping check
            if (Math.Abs(trainingLoss - previousLoss) < ConvergenceThreshold)
            {
                Console.WriteLine($"Converged at epoch {epoch} with loss {trainingLoss}");
                break;
            }

            previousLoss = trainingLoss;

            // Print progress every 100 epochs
            if (epoch % 100 == 0)
            {
                Console.WriteLine(
                    $"Epoch {epoch}: Training Loss = {trainingLoss:F6}, Validation Loss = {validationLoss:F6}");
            }
        }

        return history;
    }

    /// <summary>
    /// Calculate mean squared error loss
    /// </summary>
    private double CalculateLoss(List<double[]> inputs, List<double[]> targets, List<int> indices)
    {
        double totalLoss = 0;
        foreach (int idx in indices)
        {
            double[] prediction = Forward(inputs[idx]);
            double[] target = targets[idx];

            for (int i = 0; i < prediction.Length; i++)
            {
                double error = prediction[i] - target[i];
                totalLoss += error * error;
            }
        }

        return totalLoss / (indices.Count * targets[0].Length);
    }

    /// <summary>
    /// Backpropagation algorithm for updating weights and biases
    /// </summary>
    private void Backpropagate(double[] input, double[] target)
    {
        // Forward pass
        Forward(input);

        int numLayers = _layerSizes.Length;
        double[][] deltas = new double[numLayers][];

        for (int i = 0; i < numLayers; i++)
        {
            deltas[i] = new double[_layerSizes[i]];
        }

        // Calculate output layer deltas (using MSE derivative)
        for (int i = 0; i < _layerSizes[numLayers - 1]; i++)
        {
            double output = _neurons[numLayers - 1][i];
            deltas[numLayers - 1][i] = output - target[i];
        }

        // Calculate hidden layer deltas
        for (int l = numLayers - 2; l > 0; l--)
        {
            for (int i = 0; i < _layerSizes[l]; i++)
            {
                double sum = 0;
                for (int j = 0; j < _layerSizes[l + 1]; j++)
                {
                    sum += deltas[l + 1][j] * _weights[l][i][j];
                }

                // Apply derivative of ReLU
                double activation = _neurons[l][i];
                deltas[l][i] = activation > 0 ? sum : 0;
            }
        }

        // Update weights and biases
        for (int l = 0; l < numLayers - 1; l++)
        {
            for (int i = 0; i < _layerSizes[l]; i++)
            {
                for (int j = 0; j < _layerSizes[l + 1]; j++)
                {
                    _weights[l][i][j] -= LearningRate * _neurons[l][i] * deltas[l + 1][j];
                }
            }

            for (int j = 0; j < _layerSizes[l + 1]; j++)
            {
                _biases[l][j] -= LearningRate * deltas[l + 1][j];
            }
        }
    }

    /// <summary>
    /// ReLU activation function
    /// </summary>
    private double ReLU(double x)
    {
        return Math.Max(0, x);
    }

    /// <summary>
    /// Helper method to generate random doubles
    /// </summary>
    private double RandomDouble()
    {
        return _random.NextDouble();
    }

    /// <summary>
    /// Helper method to shuffle a list
    /// </summary>
    private void Shuffle<T>(List<T> list)
    {
        int n = list.Count;
        while (n > 1)
        {
            n--;
            int k = _random.Next(n + 1);
            T value = list[k];
            list[k] = list[n];
            list[n] = value;
        }
    }
}

public class TrainingHistory
{
    public List<int> Epochs { get; } = new List<int>();
    public List<double> TrainingLoss { get; } = new List<double>();
    public List<double> ValidationLoss { get; } = new List<double>();

    public void AddEpoch(int epoch, double trainingLoss, double validationLoss)
    {
        Epochs.Add(epoch);
        TrainingLoss.Add(trainingLoss);
        ValidationLoss.Add(validationLoss);
    }

    public void PrintSummary()
    {
        Console.WriteLine($"Training completed in {Epochs.Count} epochs");
        Console.WriteLine($"Final training loss: {TrainingLoss.Last():F6}");
        if (ValidationLoss.Count > 0)
        {
            Console.WriteLine($"Final validation loss: {ValidationLoss.Last():F6}");
        }
    }
}