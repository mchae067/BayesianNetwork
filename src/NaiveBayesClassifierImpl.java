import java.util.*;

/**
 * Your implementation of a naive bayes classifier. Please implement all four
 * methods.
 */
public class NaiveBayesClassifierImpl implements NaiveBayesClassifier {

	//Initial variables for ham and spam counts
	private Integer hams = new Integer(0);
	private Integer spams = new Integer(0);
	private Integer hamTokens = new Integer(0);
	private Integer spamTokens = new Integer(0);
	private Integer numItems;
	//delta given in prompt
	private double delta = 0.00001;
	//v given in train()
	private Integer v;

	private Map<String, Integer> hamWords = new TreeMap<String, Integer>();
	private Map<String, Integer> spamWords = new TreeMap<String, Integer>();


	/**
	 * Trains the classifier with the provided training data and vocabulary size
	 */
	@Override
	public void train(Instance[] trainingData, int v) {
		this.v = v;
		numItems = trainingData.length;

		for (Instance currentItem : trainingData)
		{
			//for hams:
			String[] currentWords = currentItem.words;
			if(currentItem.label==Label.HAM) {
				hams++;
				hamTokens+=currentWords.length;
				for (String currentWord : currentWords) {
					if (!hamWords.containsKey(currentWord)) {
						hamWords.put(currentWord, 1);
					} 
					else {
						int currentWordCount = hamWords.get(currentWord);
						hamWords.put(currentWord, currentWordCount+1);
					}
				}
			} 
			//for spams:
			else {
				spams++;
				spamTokens+=currentWords.length;
				for (String currentWord : currentWords) {
					if (!spamWords.containsKey(currentWord)) {
						spamWords.put(currentWord, 1);
					} 
					else {
						int currentWordCount = spamWords.get(currentWord);
						spamWords.put(currentWord, currentWordCount+1);
					}
				}
			}
		}
	}


	/**
	 * Returns the prior probability of the label parameter, i.e. P(SPAM)
	 * or P(HAM)
	 */
	@Override
	public double p_l(Label label) {
		if(label==Label.HAM) {
			return hams.doubleValue()/numItems.doubleValue();
		} else {
			return spams.doubleValue()/numItems.doubleValue();					
		}
	}

	
	/**
	 * Returns the smoothed conditional probability of the word given the label,
	 * i.e. P(word|SPAM) or P(word|HAM)
	 */
	@Override
	public double p_w_given_l(String word, Label label) {
		if(label==Label.HAM) {
			if (!hamWords.containsKey(word)) {
				return delta/(v.doubleValue()*delta+hamTokens);
			}
			else {
				return (hamWords.get(word).doubleValue()+delta)
						/(v.doubleValue()*delta+hamTokens);
			}
		} 
		else {
			if (!spamWords.containsKey(word)) {
				return delta/(v.doubleValue()*delta+spamTokens);
			}
			else {
				return (spamWords.get(word).doubleValue()+delta)
						/(v.doubleValue()*delta+spamTokens);
			}
		}
	}

	
	//Additional method to calculate the log probability
	private double logProbability(String[] words, Label label)
	{
		double p = Math.log(p_l(label));
		for (String currentWord : words)
			p+=Math.log(p_w_given_l(currentWord, label));
		return p;
	}
	
	/**
	 * Classifies an array of words as either SPAM or HAM. 
	 */
	@Override
	public ClassifyResult classify(String[] words) {
		ClassifyResult result = new ClassifyResult();
		double log_prob_spam = logProbability(words, Label.SPAM);
		double log_prob_ham = logProbability(words, Label.HAM);

		Label label = Label.SPAM;
		if (log_prob_ham > log_prob_spam) {
			label = Label.HAM;
		}
		result.label = label;
		result.log_prob_spam = log_prob_spam;
		result.log_prob_ham = log_prob_ham;

		return result;
	}
}
