from utils.compute_tf_jacobian_models import generate_vjp
import sys
import os

if __name__ == "__main__":
    model_root = sys.argv[1]
    decoder_path = os.path.join(model_root, "tf_models/decoder.pb")
    vjp_path = os.path.join(model_root, "tf_models/decoder_vjp.pb")
    generate_vjp(decoder_path, vjp_path)
    print("Output vjp model to:", vjp_path)
