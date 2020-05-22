# =================
# SciBERT Basevocab
# =================
for ((seed=0;seed<=40;seed+=10)); do
	# --------------
	# POOLED
	# --------------
	# No fine-tune
	outdir=results/relationship_extraction/substance_interactions_scibert_pooled_noft/${seed}
	logfile=${outdir}/LOG
	python -u relationship_extraction.py --dataset data/substance_interactions.csv \
					  --outdir ${outdir} \
					  --bert_model_file models/scibert_basevocab_cased_TF2/ \
					  --bert_model_class pooled \
					  --random_seed ${seed} \
					  --epochs 5 \
					  --mask_sentences \
					  --checkpoint_dirname model_checkpoints \
					  --tensorboard_dirname tensorboard_logs \
					  --no_finetune > results/tmp.log
	mv results/tmp.log ${logfile}

	# With fine-tune
	outdir=results/relationship_extraction/substance_interactions_scibert_pooled_ft/${seed}
	logfile=${outdir}/LOG
	python -u relationship_extraction.py --dataset data/substance_interactions.csv \
					  --outdir ${outdir} \
					  --bert_model_file models/scibert_basevocab_cased_TF2/ \
					  --bert_model_class pooled \
					  --random_seed ${seed} \
					  --epochs 5 \
					  --mask_sentences \
					  --checkpoint_dirname model_checkpoints \
					  --tensorboard_dirname tensorboard_logs > results/tmp.log
	mv results/tmp.log ${logfile}

	# --------------
	# ENTITY
	# --------------
	# No fine-tune
	outdir=results/relationship_extraction/substance_interactions_scibert_entity_noft/${seed}
	logfile=${outdir}/LOG
	python -u relationship_extraction.py --dataset data/substance_interactions.csv \
					  --outdir ${outdir} \
					  --bert_model_file models/scibert_basevocab_cased_TF2/ \
					  --bert_model_class entity \
					  --random_seed ${seed} \
					  --epochs 5 \
					  --mask_sentences \
					  --checkpoint_dirname model_checkpoints \
					  --tensorboard_dirname tensorboard_logs \
					  --no_finetune > results/tmp.log
	mv results/tmp.log ${logfile}

	# With fine-tune
	outdir=results/relationship_extraction/substance_interactions_scibert_entity_ft/${seed}
	logfile=${outdir}/LOG
	python -u relationship_extraction.py --dataset data/substance_interactions.csv \
					  --outdir ${outdir} \
					  --bert_model_file models/scibert_basevocab_cased_TF2/ \
					  --bert_model_class entity \
					  --random_seed ${seed} \
					  --epochs 5 \
					  --mask_sentences \
					  --checkpoint_dirname model_checkpoints \
					  --tensorboard_dirname tensorboard_logs > results/tmp.log
	mv results/tmp.log ${logfile}
done
