using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using CSCore.Codecs.WAV;
using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Optimizers;
using static Tensorflow.KerasApi;
using static Tensorflow.Binding;
using NumpyDotNet;

namespace CrepeSharp
{
    public static class CrepeSharp
    {
        private const int MODEL_SRATE = 16000;
        private static Dictionary<ModelCapacity, Model> models = new Dictionary<ModelCapacity, Model>()
        {
            {ModelCapacity.Tiny, null},
            {ModelCapacity.Small, null},
            {ModelCapacity.Medium, null},
            {ModelCapacity.Large, null},
            {ModelCapacity.Full, null}
        };

        /// <summary>
        /// Build the CNN model and load the weights
        /// </summary>
        /// <param name="capacity">String specifying the model capacity, which determines the model's
        ///                         capacity multiplier to 4 (tiny), 8 (small), 16 (medium), 24 (large),
        ///                         or 32 (full). 'full' uses the model size specified in the paper,
        ///                         and the others use a reduced number of filters in each convolutional
        ///                         layer, resulting in a smaller model that is faster to evaluate at the
        ///                         cost of slightly reduced pitch estimation accuracy.</param>
        /// <returns>The pre-trained keras model loaded in memory</returns>
        public static Model BuildAndLoadModel(ModelCapacity capacity)
        {
            if (models[capacity] != null) return models[capacity];
            
            var capacity_multiplier = 4 * ((int)capacity + 1);
            var layers = new[] {1, 2, 3, 4, 5, 6};
            var filters = new[] { 32, 4, 4, 4, 8, 16 };
            for (int i = 0; i < filters.Length; i++)
            {
                filters[i] *= capacity_multiplier;
            }
            var widths = new[] {512, 64, 64, 64, 64, 64};
            var strides = new[] {(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)};
            
            var x = keras.Input(shape: (1024), name: "input", dtype: TF_DataType.TF_FLOAT);
            var y = tf.reshape(tensor: x, shape: (1024, 1, 1), name: "input-reshape");

            // todo: make sure layers length doesn't change
            for (int i = 0; i < layers.Length; i++)
            {
                var l = layers[i];
                var f = filters[i];
                var w = widths[i];
                var s = strides[i];

                keras.layers.Conv2D(f, (w, 1), strides: s, padding: "same", activation: "relu").Apply(y);
                keras.layers.BatchNormalization().Apply(y);
                keras.layers.MaxPooling2D(pool_size: (2, 1), strides: null, padding: "valid").Apply(y);
                keras.layers.Dropout(rate: 0.25f).Apply(y);
            }

            keras.layers.Permute(new []{2,1,3}).Apply(y);
            keras.layers.Flatten().Apply(y);
            keras.layers.Dense(360, activation: "sigmoid");

            var model = keras.Model(x, y);

            var package_dir = Directory.GetCurrentDirectory();
            var filename = $"model-{capacity.ToString()}.h5";
            
            try
            {
                model.load_weights($"{package_dir}{filename}");
            }
            catch (Exception e)
            {
                Console.WriteLine(e);
                throw;
            }

            model.compile(optimizer: new Adam(), loss: new BinaryCrossentropy());
            //model.compile(new Adam()); // TODO: binary crossentropy needed

            models[capacity] = model;

            return model;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="file"></param>
        /// <param name="suffix"></param>
        /// <param name="output_dir"></param>
        /// <returns>The output path of an output file corresponding to a wav file</returns>
        public static string OutputPath(string file, string suffix, string output_dir)
        {
            var path = Regex.Replace(file, @"(?i).wav$", suffix);
            if (output_dir != "")
            {
                path = Path.Join(output_dir, new DirectoryInfo(path).Name); // idk how i feel about this
            }

            return path;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="audio">The audio samples. Multichannel audio will be downmixed.</param>
        /// <param name="sr">Sample rate of the audio samples. The audio will be resampled if
        /// the sample rate is not 16 kHz, which is expected by the model.</param>
        /// <param name="model_capacity">Model capacity</param>
        /// <param name="center">If `True` (default), the signal `audio` is padded so that frame
        /// `D[:, t]` is centered at `audio[t * hop_length]` If `False`,
        /// then `D[:, t]` begins at `audio[t * hop_length]`</param>
        /// <param name="step_size">The step size in milliseconds for running pitch estimation.</param>
        /// <param name="verbose">Set the keras verbosity mode: 1 (default) will print out a progress bar
        /// during prediction, 0 will suppress all non-error printouts.</param>
        /// <returns>The raw activation matrix</returns>
        public static ndarray GetActivation(
            ndarray audio, int sr = MODEL_SRATE, ModelCapacity model_capacity = ModelCapacity.Full, 
            bool center = true, int step_size = 10,  int verbose = 1) // i know this is quite long
        {
            var model = BuildAndLoadModel(model_capacity);

            if (len(audio.shape) == 2) audio = np.mean(1); // hmm
            audio = audio.astype(np.Float32);
            //if (sr != MODEL_SRATE) resample
            if (center)
            {
                // todo: check paddings
            }

            var hop_length = Convert.ToInt32(MODEL_SRATE * step_size / 1000);
            var n_frames = 1 + Convert.ToInt32(len(audio) / hop_length);
            var frames = np.as_strided(x: audio, shape: (1024, n_frames).GetShape(),
                strides: (audio.ItemSize, hop_length * audio.ItemSize).GetShape()); // todo:: check this
            frames = frames.Transpose().Copy();
            
            frames -= np.mean(frames, axis: 1)[':', np.newaxis];
            frames /= np.clip(a: np.std(frames, axis: 1)[':', np.newaxis], a_min: 1e-8, a_max: null);

            return model.predict(tf.convert_to_tensor(frames, TF_DataType.TF_FLOAT), verbose: verbose);
        }

        public static (ndarray, ndarray, ndarray, ndarray) Predict(ndarray audio, int sr = MODEL_SRATE,
            ModelCapacity model_capacity = ModelCapacity.Full,
            bool viterbi = false, bool center = true, int step_size = 10, int verbose = 1)
        {
            var activation = GetActivation(audio, sr, model_capacity, center, step_size, verbose);
            var confidence = np.max(activation, axis: 1);

            ndarray cents;

            ndarray frequency = 10 * 2 ** (cents / 1200);
            frequency[np.isnan(frequency)] = 0;

            var time = np.arange(confidence.shape[0]) * step_size / 1000.0;

            return (time, frequency, confidence, activation);
        }
        
        public static 

        public static void ProcessFile(string file, string output = null,
            ModelCapacity model_capacity = ModelCapacity.Full,
            bool viterbi = false, bool center = true, bool save_activation = false, bool save_plot = false,
            bool plot_voicing = false, int step_size = 10, bool verbose = true)
        {
            var reader = new WaveFileReader(file);
            
        }
    }
}