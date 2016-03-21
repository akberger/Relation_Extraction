
#pipeling for training maxent model, classifying test/devset
#and evaluating accuracy

#Usage: bash relationExtractionPipeline.sh <train_corpus> <test_corpus> <model_name> <gold_file_for_test> <test_comment> <README.md path>
#ex: bash bin/relationExtractionPipeline.sh gold/rel-trainset.gold raw/rel-devset.raw postagged-files/ models/rel-model3_btwn_prv gold/rel-devset.gold "features: added between_words and prev_words" README.md 


TRAIN=$1
TRAIN_FEATS="${TRAIN}_features.txt"
TEST=$2
TEST_FEATS="${TEST}_features.txt"
TOKENS=$3
MODEL=$4
GOLD=$5
MESSAGE=$6
README=$7

echo "Extracting features from training"
python /home/g/grad/cmward/Relation_Extraction/ExtractRelationFeatures.py $TRAIN $TRAIN_FEATS $TOKENS train
echo ""
#exit
echo "Extracting features from test"
python /home/g/grad/cmward/Relation_Extraction/ExtractRelationFeatures.py $TEST $TEST_FEATS $TOKENS
echo ""
#exit
echo "Training MaxEnt classifier..."
bash /home/j/clp/chinese/bin/mallet-maxent-classifier.sh -train -model=$MODEL -gold=$TRAIN_FEATS
echo ""

TAGGED="${TEST}.tagged"
echo "Classifying..."
bash /home/j/clp/chinese/bin/mallet-maxent-classifier.sh -classify -model=$MODEL -input=$TEST_FEATS > $TAGGED
echo ""

echo "Computing accuracy..."
echo $MESSAGE >> $README
OUT=$(python /home/j/xuen/teaching/cosi137/spring-2015/projects/project3/relation-evaluator.py $GOLD $TAGGED 2>&1)
echo $OUT
echo $OUT >> "${README}"
echo "" >> "${README}"
