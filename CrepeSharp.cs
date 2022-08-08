using System;
using Tensorflow;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;

namespace CrepeSharp
{
    public static class CrepeSharp
    {
        private const int MODEL_SRATE = 16000;

        public static Model BuildAndLoadModel(ModelCapacity capacity)
        {
            var capacity_multiplier = 4 * ((int)capacity + 1);
            var layers = new[] {1, 2, 3, 4, 5, 6};

            var widths = new[] {512, 64, 64, 64, 64, 64};
            var strides = new[] {(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)};

            var inputLayerArgs = new InputLayerArgs
            {
                InputShape = (1024),
                Name = "input",
                DType = TF_DataType.TF_FLOAT
            };

            var reshapeArgs = new ReshapeArgs
            {
                TargetShape = (1024, 1, 1),
                Name = "input-reshape"
            };

            var x = new InputLayer(inputLayerArgs);
            var y = new Reshape(reshapeArgs);
 
        }
    }
}