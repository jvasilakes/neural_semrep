# ==============
# BioBERT v1.1
# ==============
# 5 runs with different random seeds: 0, 10, 20, 30, 40
model=biobert
dataset=data/semrep_gold_standard/adjudicated.csv
epoch=4
prediction_outfile=predictions_semrep.csv
for ((seed=0;seed<=40;seed+=10)); do
	# --------------
	# POOLED
	# --------------
	# No fine-tune
	#outdir=results/relationship_extraction/substance_interactions_${model}_pooled_noft/${seed}
	#weights_file=$(ls -1 ${outdir}/model_checkpoints/weights.0${epoch}-*)
	#labels="NULL STIMULATES INHIBITS INTERACTS_WITH"
	#echo "==================================" >> results/tmp.log
	#echo "Predicting on: ${dataset}" >> results/tmp.log
	#echo "Labels: ${labels}" >> results/tmp.log
	#echo "Weights file: ${weights_file}" >> results/tmp.log
	#python -u run_prediction.py --dataset ${dataset} \
	#			    --outfile ${outdir}/${prediction_outfile} \
	#		   	    --bert_model_class pooled \
	#			    --bert_weights_file ${weights_file} \
	#			    --bert_config_file ${outdir}/PooledModel.json \
	#			    --classes ${labels} >> results/tmp.log 2> results/tmp.err
	#echo -e "==================================\n" >> results/tmp.log
	#mv results/tmp.log ${outdir}/predict.log
	#mv results/tmp.err ${outdir}/predict.err

	# With fine-tune
	outdir=results/relationship_extraction/substance_interactions_${model}_pooled_ft/${seed}
	weights_file=$(ls -1 ${outdir}/model_checkpoints/weights.0${epoch}-*)
	labels="NULL STIMULATES INHIBITS INTERACTS_WITH"
	echo "==================================" >> results/tmp.log
	echo "Predicting on: ${dataset}" >> results/tmp.log
	echo "Labels: ${labels}" >> results/tmp.log
	echo "Weights file: ${weights_file}" >> results/tmp.log
	python -u run_prediction.py --dataset ${dataset} \
				    --outfile ${outdir}/${prediction_outfile} \
			   	    --bert_model_class pooled \
				    --bert_weights_file ${weights_file} \
				    --bert_config_file ${outdir}/PooledModel.json \
				    --classes ${labels} >> results/tmp.log 2> results/tmp.err
	echo -e "==================================\n" >> results/tmp.log
	mv results/tmp.log ${outdir}/predict.log
	mv results/tmp.err ${outdir}/predict.err

	# --------------
	# ENTITY
	# --------------
	# No fine-tune
	outdir=results/relationship_extraction/substance_interactions_${model}_entity_noft/${seed}
	weights_file=$(ls -1 ${outdir}/model_checkpoints/weights.0${epoch}-*)
	labels="NULL STIMULATES INHIBITS INTERACTS_WITH"
	echo "==================================" >> results/tmp.log
	echo "Predicting on: ${dataset}" >> results/tmp.log
	echo "Labels: ${labels}" >> results/tmp.log
	echo "Weights file: ${weights_file}" >> results/tmp.log
	python -u run_prediction.py --dataset ${dataset} \
				    --outfile ${outdir}/${prediction_outfile} \
			   	    --bert_model_class entity \
				    --bert_weights_file ${weights_file} \
				    --bert_config_file ${outdir}/EntityModel.json \
				    --classes ${labels} >> results/tmp.log 2> results/tmp.err
	echo -e "==================================\n" >> results/tmp.log
	mv results/tmp.log ${outdir}/predict.log
	mv results/tmp.err ${outdir}/predict.err

	# With fine-tune
	outdir=results/relationship_extraction/substance_interactions_${model}_entity_ft/${seed}
	weights_file=$(ls -1 ${outdir}/model_checkpoints/weights.0${epoch}-*)
	labels="NULL STIMULATES INHIBITS INTERACTS_WITH"
	echo "==================================" >> results/tmp.log
	echo "Predicting on: ${dataset}" >> results/tmp.log
	echo "Labels: ${labels}" >> results/tmp.log
	echo "Weights file: ${weights_file}" >> results/tmp.log
	python -u run_prediction.py --dataset ${dataset} \
				    --outfile ${outdir}/${prediction_outfile} \
			   	    --bert_model_class entity \
				    --bert_weights_file ${weights_file} \
				    --bert_config_file ${outdir}/EntityModel.json \
				    --classes ${labels} >> results/tmp.log 2> results/tmp.err
	echo -e "==================================\n" >> results/tmp.log 
	mv results/tmp.log ${outdir}/predict.log
	mv results/tmp.err ${outdir}/predict.err
done
