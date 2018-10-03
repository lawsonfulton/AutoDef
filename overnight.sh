#! /bin/bash
python scripts/unified_gen_and_train.py configs/unified/X.json models/search/X --retrain
python scripts/unified_gen_and_train.py configs/unified/dragon.json models/search/dragon --retrain
python scripts/unified_gen_and_train.py configs/unified/armadillo.json models/search/armadillo --retrain
python scripts/unified_gen_and_train.py configs/unified/catapult.json models/search/catapult --retrain
python scripts/unified_gen_and_train.py configs/unified/mounted_bar.json models/search/mounted_bar --retrain
