# ===================
# SciBERT Basevocab
# ===================
datadir=data/substance_interactions_splits/
model=models/scibert_basevocab_cased_TF2/
num_epochs=4
for ((seed=0;seed<=40;seed+=10)); do
	# --------------
	# POOLED
	# --------------
	# No fine-tune
	#outdir=results/relationship_extraction/substance_interactions_scibert_pooled_noft/${seed}
	#logfile=${outdir}/LOG
	#python -u relationship_classification.py --datadir ${datadir} \
	#			 	         --outdir ${outdir} \
	#				         --random_seed ${seed} \
	#				         --bert_model_class pooled \
	#				         --bert_model_file ${model} \
	#				         --epochs ${num_epochs} \
	#				         --checkpoint_dirname model_checkpoints \
	#				         --tensorboard_dirname tensorboard_logs \
	#				         --no_finetune \
	#					 --verbose 2 > results/tmp.log
	#mv results/tmp.log ${logfile}

	# With fine-tune
	outdir=results/relationship_extraction/substance_interactions_scibert_pooled_ft/${seed}
	logfile=${outdir}/LOG
	python -u relationship_classification.py --datadir ${datadir} \
					         --outdir ${outdir} \
					         --random_seed ${seed} \
					         --bert_model_class pooled \
					         --bert_model_file ${model} \
					         --epochs ${num_epochs} \
					         --checkpoint_dirname model_checkpoints \
					         --tensorboard_dirname tensorboard_logs \
						 --verbose 2 > results/tmp.log
	mv results/tmp.log ${logfile}

	# --------------
	# ENTITY
	# --------------
	# No fine-tune
	outdir=results/relationship_extraction/substance_interactions_scibert_entity_noft/${seed}
	logfile=${outdir}/LOG
	python -u relationship_classification.py --datadir ${datadir} \
					         --outdir ${outdir} \
					         --random_seed ${seed} \
					         --bert_model_class entity \
					         --bert_model_file ${model} \
					         --epochs ${num_epochs} \
					         --checkpoint_dirname model_checkpoints \
					         --tensorboard_dirname tensorboard_logs \
					         --no_finetune \
						 --verbose 2 > results/tmp.log
	mv results/tmp.log ${logfile}

	# With fine-tune
	outdir=results/relationship_extraction/substance_interactions_scibert_entity_ft/${seed}
	logfile=${outdir}/LOG
	python -u relationship_classification.py --datadir ${datadir} \
					         --outdir ${outdir} \
					         --random_seed ${seed} \
					         --bert_model_class entity \
					         --bert_model_file ${model} \
					         --epochs ${num_epochs} \
					         --checkpoint_dirname model_checkpoints \
					         --tensorboard_dirname tensorboard_logs \
						 --verbose 2 > results/tmp.log
	mv results/tmp.log ${logfile}
done
