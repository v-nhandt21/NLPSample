if [[( $1 == "train")]];
then

#
if [[( $2 == "sequence")]];
then

##
if [[( $3 == "bert")]];
then
python Sequence_Labeling/BERT/${1}.py 
fi
##
if [[( $3 == "bilstm")]];
then
python Sequence_Labeling/BiLSTM/${1}.py
fi
fi

#
if [[( $2 == "sentiment")]];
then

##
if [[( $3 == "rnn")]];
then
python Sentiment_analysis/Glove/{1}.py rnn
fi

##
if [[( $3 == "lstm")]];
then
python Sentiment_analysis/Glove/{1}.py lstm
fi

##
if [[( $3 == "cnn")]];
then
python Sentiment_analysis/Glove/{1}.py cnn 
fi

##
if [[( $3 == "gru")]];
then
python Sentiment_analysis/Glove/{1}.py gru
fi


##
if [[( $3 == "att_dot")]];
then
python Sentiment_analysis/Glove/{1}.py att_dot
fi


##
if [[( $3 == "att_mul")]];
then
python Sentiment_analysis/Glove/{1}.py att_mul
fi

##
if [[( $3 == "att_add")]];
then
python Sentiment_analysis/Glove/{1}.py att_add
fi

##
if [[( $3 == "bert")]];
then
python Sentiment_analysis/Glove/{1}.py bert
fi

##
if [[( $3 == "att_bilstm_add")]];
then
python Sentiment_analysis/Glove/{1}.py att_bilstm_add
fi

fi
fi
