using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace HousingConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            // Load data
            var data = context.Data.LoadFromTextFile<HousingData>("./data/housing.csv", hasHeader: true,
                separatorChar: ',');

            // Part of data should be separated for training purposes
            var split = context.Data.TrainTestSplit(data, testFraction: 0.2);
            
            var features = split.TrainSet.Schema
                .Select(col => col.Name)
                .Where(colName => colName != "Label" && colName != "OceanProximity")
                .ToArray();

            var pipeline = context.Transforms.Text.FeaturizeText("Text", "OceanProximity")
                .Append(context.Transforms.Concatenate("Features", features))
                .Append(context.Transforms.Concatenate("Feature", "Features", "Text"))
                .Append(context.Regression.Trainers.LbfgsPoissonRegression());

            var model = pipeline.Fit(split.TrainSet);

            var predictions = model.Transform(split.TestSet);

            var metrics = context.Regression.Evaluate(predictions);

            Console.WriteLine($"R^2 - {metrics.RSquared}");
        }
    }

    internal class HousingData
    {
        [LoadColumn(0)]
        public float Longitude { get; set; }

        [LoadColumn(1)]
        public float Latitude { get; set; }

        [LoadColumn(2)]
        public float HousingMedianAge { get; set; }

        [LoadColumn(3)]
        public float TotalRooms { get; set; }

        [LoadColumn(4)]
        public float TotalBedrooms { get; set; }

        [LoadColumn(5)]
        public float Population { get; set; }

        [LoadColumn(6)]
        public float Households { get; set; }

        [LoadColumn(7)]
        public float MedianIncome { get; set; }

        [LoadColumn(8), ColumnName("Label")]
        public float MedianHouseValue { get; set; }

        [LoadColumn(9)]
        public string OceanProximity { get; set; }
    }
}
