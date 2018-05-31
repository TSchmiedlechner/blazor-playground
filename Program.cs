using System;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML;
using System.Threading.Tasks;

namespace mlnet
{
  class Program
  {
    const string _datapath = @"./data/taxi-fare-train.csv";
    const string _testdatapath = @"./data/taxi-fare-test.csv";
    const string _modelpath = @"./data/Model.zip";

    static async Task Main(string[] args)
    {
      PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = await Train();
      Evaluate(model);
    }

    public static async Task<PredictionModel<TaxiTrip, TaxiTripFarePrediction>> Train()
    {
      var pipeline = new LearningPipeline
      {
        new TextLoader<TaxiTrip>(_datapath, useHeader: true, separator: ","),
        new ColumnCopier(("fare_amount", "Label")),
        new CategoricalOneHotVectorizer("vendor_id",
                                        "rate_code",
                                        "payment_type"),
        new ColumnConcatenator("Features",
                               "vendor_id",
                               "rate_code",
                               "passenger_count",
                               "trip_distance",
                               "payment_type"),
        new FastTreeRegressor()
      };
      var model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();
      await model.WriteAsync(_modelpath);
      return model;
    }

    private static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
    {
      var testData = new TextLoader<TaxiTrip>(_testdatapath, useHeader: true, separator: ",");
      var evaluator = new RegressionEvaluator();
      RegressionMetrics metrics = evaluator.Evaluate(model, testData);

      // Rms should be around 2.795276
      Console.WriteLine("Rms=" + metrics.Rms);
      Console.WriteLine("RSquared = " + metrics.RSquared);
    }
  }
}
