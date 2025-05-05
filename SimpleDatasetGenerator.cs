using Simple_neural_network.model;

namespace Simple_neural_network;

public class SimpleDatasetGenerator
{
    private readonly Random _random = new Random();

    public List<DataPoint> Generate(int sampleCount, double min = -10, double max = 10)
    {
        var dataset = new List<DataPoint>();

        for (int i = 0; i < sampleCount; i++)
        {
            double x = RandomDouble(min, max);
            double y = RandomDouble(min, max);
            double z = x + 2 * y;

            dataset.Add(new DataPoint
            {
                X = x,
                Y = y,
                Z = z
            });
        }

        return dataset;
    }

    private double RandomDouble(double min, double max)
    {
        return min + (_random.NextDouble() * (max - min));
    }
}