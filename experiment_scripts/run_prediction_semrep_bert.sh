# ==============
# BERT Base
# ==============
# 5 runs with different random seeds: 0, 10, 20, 30, 40
for ((seed=0;seed<=40;seed+=10)); do
	# --------------
	# POOLED
	# --------------
	# No fine-tune
	outdir=results/relationship_extraction/substance_interactions_bert_pooled_noft/${seed}
	logfile=${outdir}/predict.LOG
	weights_file=$(ls -1 ${outdir}/model_checkpoints/weights.04-*)
	prediction_file=predictions_semrep.csv
	labels="NULL STIMULATES INHIBITS INTERACTS_WITH"
	echo "==================================" >> results/tmp.log
	echo "Predicting on: ${prediction_file}" >> results/tmp.log
	echo "Labels: ${labels}" >> results/tmp.log
	echo "Weights file: ${weights_file}" >> results/tmp.log
	python -u run_prediction.py --dataset data/semrep_gold_standard/adjudicated.csv \
				    --outfile ${outdir}/${prediction_file} \
			   	    --bert_model_class pooled \
				    --bert_weights_file ${weights_file} \
				    --bert_config_file ${outdir}/PooledModel.json \
				    --classes ${labels} >> results/tmp.log 2> results/tmp.err
	echo "==================================\n" >> results/tmp.log
	mv results/tmp.log ${logfile}

	# With fine-tune
	outdir=results/relationship_extraction/substance_interactions_bert_pooled_ft/${seed}
	logfile=${outdir}/predict.LOG
	weights_file=$(ls -1 ${outdir}/model_checkpoints/weights.04-*)
	prediction_file=predictions_semrep.csv
	labels="NULL STIMULATES INHIBITS INTERACTS_WITH"
	echo "==================================" >> results/tmp.log
	echo "Predicting on: ${prediction_file}" >> results/tmp.log
	echo "Labels: ${labels}" >> results/tmp.log
	echo "Weights file: ${weights_file}" >> results/tmp.log
	python -u run_prediction.py --dataset data/semrep_gold_standard/adjudicated.csv \
				    --outfile ${outdir}/${prediction_file} \
			   	    --bert_model_class pooled \
				    --bert_weights_file ${weights_file} \
				    --bert_config_file ${outdir}/PooledModel.json \
				    --classes ${labels} >> results/tmp.log
	echo "==================================\n" >> results/tmp.log
	mv results/tmp.log ${logfile}

	# --------------
	# ENTITY
	# --------------
	# No fine-tune
	outdir=results/relationship_extraction/substance_interactions_bert_entity_noft/${seed}
	logfile=${outdir}/predict.LOG
	weights_file=$(ls -1 ${outdir}/model_checkpoints/weights.04-*)
	prediction_file=predictions_semrep.csv
	labels="NULL STIMULATES INHIBITS INTERACTS_WITH"
	echo "==================================" >> results/tmp.log
	echo "Predicting on: ${prediction_file}" >> results/tmp.log
	echo "Labels: ${labels}" >> results/tmp.log
	echo "Weights file: ${weights_file}" >> results/tmp.log
	python -u run_prediction.py --dataset data/semrep_gold_standard/adjudicated.csv \
				    --outfile ${outdir}/${prediction_file} \
			   	    --bert_model_class entity \
				    --bert_weights_file ${weights_file} \
				    --bert_config_file ${outdir}/EntityModel.json \
				    --classes ${labels} >> results/tmp.log
	echo "==================================\n" >> results/tmp.log
	mv results/tmp.log ${logfile}

	# With fine-tune
	outdir=results/relationship_extraction/substance_interactions_bert_entity_ft/${seed}
	logfile=${outdir}/predict.LOG
	weights_file=$(ls -1 ${outdir}/model_checkpoints/weights.04-*)
	prediction_file=predictions_semrep.csv
	labels="NULL STIMULATES INHIBITS INTERACTS_WITH"
	echo "==================================" >> results/tmp.log
	echo "Predicting on: ${prediction_file}" >> results/tmp.log
	echo "Labels: ${labels}" >> results/tmp.log
	echo "Weights file: ${weights_file}" >> results/tmp.log
	python -u run_prediction.py --dataset data/semrep_gold_standard/adjudicated.csv \
				    --outfile ${outdir}/${prediction_file} \
			   	    --bert_model_class entity \
				    --bert_weights_file ${weights_file} \
				    --bert_config_file ${outdir}/EntityModel.json \
				    --classes ${labels} >> results/tmp.log
	echo "==================================\n" >> results/tmp.log
	mv results/tmp.log ${logfile}
done
