using Tensorflow;
using Tensorflow.Keras.Losses;
using static Tensorflow.Binding;

namespace CrepeSharp
{
    public class BinaryCrossentropy : LossFunctionWrapper, ILossFunc
    {
        public BinaryCrossentropy(string reduction = "auto", string name = null, bool from_logits = false) 
            : base(reduction, name, from_logits)
        {
        }

        // todo: does this even work right??
        public override Tensor Apply(Tensor x, Tensor y, bool from_logits = false, int axis = -1)
        {
            var shape = tf.reduce_prod(tf.shape(x));
            var count = tf.cast(shape, TF_DataType.TF_FLOAT);
            x = tf.clip_by_value(x, 1e-6f, 1.0f - 1e-6f);
            var z = y * tf.log(x) + (1 - y) * tf.log(1 - x);
            var result = -1.0f / count * tf.reduce_sum(z);
            return result;
        }
        
    }
}