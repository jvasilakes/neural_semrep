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
	logfile=${outdir}/LOG
	python -u relationship_classification.py --datadir data/substance_interactions_splits/ \
					         --outdir ${outdir} \
					         --random_seed ${seed} \
					         --bert_model_class pooled \
					         --bert_model_file models/bert_en_cased_L-12_H-768_A-12/ \
					         --epochs 4 \
					         --checkpoint_dirname model_checkpoints \
					         --tensorboard_dirname tensorboard_logs \
					         --no_finetune \
					         --verbose 2 > results/tmp.log
	mv results/tmp.log ${logfile}

	# With fine-tune
	outdir=results/relationship_extraction/substance_interactions_bert_pooled_ft/${seed}
	logfile=${outdir}/LOG
	python -u relationship_classification.py --datadir data/substance_interactions_splits/ \
					         --outdir ${outdir} \
					         --random_seed ${seed} \
					         --bert_model_class pooled \
					         --bert_model_file models/bert_en_cased_L-12_H-768_A-12/ \
					         --epochs 4 \
					         --checkpoint_dirname model_checkpoints \
					         --tensorboard_dirname tensorboard_logs \
					         --verbose 2 > results/tmp.log
	mv results/tmp.log ${logfile}

	# --------------
	# ENTITY
	# --------------
	# No fine-tune
	outdir=results/relationship_extraction/substance_interactions_bert_entity_noft/${seed}
	logfile=${outdir}/LOG
	python -u relationship_classification.py --datadir data/substance_interactions_splits/ \
					         --outdir ${outdir} \
					         --random_seed ${seed} \
					         --bert_model_class entity \
					         --bert_model_file models/bert_en_cased_L-12_H-768_A-12/ \
					         --epochs 4 \
					         --checkpoint_dirname model_checkpoints \
					         --tensorboard_dirname tensorboard_logs \
					         --no_finetune \
					         --verbose 2 > results/tmp.log
	mv results/tmp.log ${logfile}

	# With fine-tune
	outdir=results/relationship_extraction/substance_interactions_bert_entity_ft/${seed}
	logfile=${outdir}/LOG
	python -u relationship_classification.py --datadir data/substance_interactions_splits/ \
					         --outdir ${outdir} \
					         --random_seed ${seed} \
					         --bert_model_class entity \
					         --bert_model_file models/bert_en_cased_L-12_H-768_A-12/ \
					         --epochs 4 \
					         --checkpoint_dirname model_checkpoints \
					         --tensorboard_dirname tensorboard_logs \
					         --verbose 2 > results/tmp.log
	mv results/tmp.log ${logfile}
done
