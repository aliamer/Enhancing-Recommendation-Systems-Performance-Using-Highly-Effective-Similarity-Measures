package net.hudup.alg.cf;

import java.rmi.RemoteException;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

import net.hudup.core.Constants;
import net.hudup.core.Util;
import net.hudup.core.alg.cf.NeighborCF;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Dataset;
import net.hudup.core.data.Profile;
import net.hudup.core.data.RatingVector;
import net.hudup.core.logistic.DSUtil;
import net.hudup.core.logistic.Inspector;
import net.hudup.core.logistic.NextUpdate;
import net.hudup.core.logistic.Vector2;
import net.hudup.core.parser.TextParserUtil;
import net.hudup.data.DocumentVector;
import net.hudup.evaluate.ui.EvaluateGUI;

/**
 * This class sets up an advanced version of neighbor collaborative filtering (Neighbor CF) algorithm with more similarity measures.
 * <br>
 * There are many authors who contributed measure to this class.<br>
 * Authors Haifeng Liu, Zheng Hu, Ahmad Mian, Hui Tian, Xuzhen Zhu contributed PSS measures and NHSM measure.<br>
 * Authors Bidyut Kr. Patra, Raimo Launonen, Ville Ollikainen, Sukumar Nandi contributed BC and BCF measures.<br>
 * Author Hyung Jun Ahn contributed PIP measure.<br>
 * Authors Keunho Choi and Yongmoo Suh contributed PC measure.<br>
 * Authors Suryakant and Tripti Mahara contributed MMD measure and CjacMD measure.<br>
 * Authors Junmei Feng, Xiaoyi Fengs, Ning Zhang, and Jinye Peng contributed Feng model.<br>
 * Authors Yi Mua, Nianhao Xiao, Ruichun Tang, Liang Luo, and Xiaohan Yin contributed Mu measure.<br>
 * Authors Yung-Shen Lin, Jung-Yi Jiang, Shie-Jue Lee contributed SMTP measure.<br>
 * Author Ali Amer contributed Amer and Amer2 measures.<br>
 * Author Loc Nguyen contributed TA (triangle area) measure.<br>
 * Authors Ali Amer and Loc Nguyen contributed quasi-TfIdf measure. Quasi-TfIdf measure is an extension of Amer2 measure and the ideology of TF and IDF.<br>
 * Author Ali Amer contributed numerical nearby similarity measure (MMNS).
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class NeighborCFExt extends NeighborCF {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Name of PSS measure.
	 */
	public static final String PSS = "pss";

	
	/**
	 * Name of NHSM measure.
	 */
	public static final String NHSM = "nhsm";

	
	/**
	 * Name of BCF measure.
	 */
	public static final String BCF = "bcf";

	
	/**
	 * Name of BCFJ measure (BCF + Jaccard).
	 */
	public static final String BCFJ = "bcfj";

	
	/**
	 * Name of SRC measure.
	 */
	public static final String SRC = "src";

	
	/**
	 * Name of PIP measure.
	 */
	public static final String PIP = "pip";

	
	/**
	 * Name of PC measure.
	 */
	public static final String PC = "pc";

	
	/**
	 * Name of MMD measure.
	 */
	public static final String MMD = "mmd";

	
	/**
	 * Name of CjacMD measure which is developed by Suryakant and Tripti Mahara.
	 */
	public static final String CJACMD = "mmd";

	
	/**
	 * Name of Feng measure.
	 */
	public static final String FENG = "feng";

	
	/**
	 * Name of Mu measure.
	 */
	public static final String MU = "mu";

	
	/**
	 * Name of SMTP measure.
	 */
	public static final String SMTP = "smtp";

	
	/**
	 * Name of Amer measure.
	 */
	public static final String AMER = "amer";

	
	/**
	 * Name of Amer2 measure.
	 */
	public static final String AMER2 = "amer2";

	
	/**
	 * Name of Amer2 + Jaccard measure.
	 */
	public static final String AMER2J = "amer2j";

	
	/**
	 * Name of Quasi-TfIdf measure.
	 */
	public static final String QUASI_TFIDF = "qti";

	
	/**
	 * Name of Quasi-TfIdf + Jaccard measure.
	 */
	public static final String QUASI_TFIDF_JACCARD = "qtij";

	
	/**
	 * Name of triangle area measure.
	 */
	public static final String TA = "ta";

	
	/**
	 * Name of triangle area + Jaccard measure.
	 */
	public static final String TAJ = "taj";

	
	/**
	 * Name of Coco measure.
	 */
	public static final String COCO = "coco";

	
	/**
	 * Name of numerical nearby similarity measure (MMNS).
	 */
	public static final String NNMS = "mmns";

	
	/**
	 * Value bins.
	 */
	public static final String VALUE_BINS_FIELD = "value_bins";

	
	/**
	 * Default value bins.
	 */
	public static final String VALUE_BINS_DEFAULT = "1, 2, 3, 4, 5";

	
	/**
	 * BCF median mode.
	 */
	public static final String BCF_MEDIAN_MODE_FIELD = "bcf_median";

	
	/**
	 * Default BCF median mode.
	 */
	public static final boolean BCF_MEDIAN_MODE_DEFAULT = true;

	
	/**
	 * Mu alpha field.
	 */
	public static final String MU_ALPHA_FIELD = "mu_alpha";

	
	/**
	 * Default Mu alpha.
	 */
	public static final double MU_ALPHA_DEFAULT = 0.5;

	
	/**
	 * Name of lambda field.
	 */
	public static final String SMTP_LAMBDA_FIELD = "smtp_lambda";

	
	/**
	 * Default lambda field.
	 */
	public static final double SMTP_LAMBDA_DEFAULT = 0.5;

	
	/**
	 * Name of general variance field.
	 */
	public static final String SMTP_GENERAL_VAR_FIELD = "smtp_general_var";

	
	/**
	 * Default general variance field.
	 */
	public static final boolean SMTP_GENERAL_VAR_DEFAULT = false;

	
	/**
	 * TA normalized mode.
	 */
	public static final String TA_NORMALIZED_FIELD = "ta_normalized";

	
	/**
	 * Default TA normalized mode.
	 */
	public static final boolean TA_NORMALIZED_DEFAULT = false;

	
	/**
	 * Value bins.
	 */
	protected List<Double> valueBins = Util.newList();
	
	
	/**
	 * Rank bins.
	 */
	protected Map<Double, Integer> rankBins = Util.newMap();
	
	
	/**
	 * Column module (column vector length) cache.
	 */
	protected Map<Integer, Object> bcfColumnModuleCache = Util.newMap();

	
	/**
	 * Default constructor.
	 */
	public NeighborCFExt() {
		// TODO Auto-generated constructor stub
	}


	@Override
	public synchronized void setup(Dataset dataset, Object...params) throws RemoteException {
		// TODO Auto-generated method stub
		super.setup(dataset, params);
		
		this.valueBins = extractConfigValueBins();
		this.rankBins = convertValueBinsToRankBins(this.valueBins);
	}


	@Override
	public synchronized void unsetup() throws RemoteException {
		// TODO Auto-generated method stub
		super.unsetup();
		
		this.rankBins.clear();
		this.valueBins.clear();
		
		this.bcfColumnModuleCache.clear();
	}


	@Override
	public List<String> getSupportedMeasures() {
		// TODO Auto-generated method stub
		List<String> measures = super.getSupportedMeasures();
		Set<String> mSet = Util.newSet();
		mSet.addAll(measures);
		mSet.add(PSS);
//		mSet.add(NHSM);
		mSet.add(BCF);
//		mSet.add(BCFJ);
		mSet.add(SRC);
		mSet.add(PIP);
		mSet.add(PC);
		mSet.add(MMD);
//		mSet.add(CJACMD);
		mSet.add(SMTP);
		mSet.add(AMER);
		mSet.add(AMER2);
//		mSet.add(AMER2J);
		mSet.add(QUASI_TFIDF);
//		mSet.add(QUASI_TFIDF_JACCARD);
		mSet.add(TA);
//		mSet.add(TAJ);
		mSet.add(COCO);
		mSet.add(NNMS);
		
		measures.clear();
		measures.addAll(mSet);
		Collections.sort(measures);
		return measures;
	}


	/**
	 * Checking whether the similarity measure requires to declare discrete bins in configuration ({@link #VALUE_BINS_FIELD}).
	 * @return true if the similarity measure requires to declare discrete bins in configuration ({@link #VALUE_BINS_FIELD}). Otherwise, return false.
	 */
	public boolean requireDiscreteRatingBins() {
		return requireDiscreteRatingBins(getMeasure());
	}
	
	
	/**
	 * Given specified measure, checking whether the similarity measure requires to declare discrete bins in configuration ({@link #VALUE_BINS_FIELD}).
	 * @param measure specified measure.
	 * @return true if the similarity measure requires to declare discrete bins in configuration ({@link #VALUE_BINS_FIELD}). Otherwise, return false.
	 */
	protected boolean requireDiscreteRatingBins(String measure) {
		if (measure == null)
			return false;
		else if (measure.equals(BCF) || measure.equals(BCFJ) ||  measure.equals(MMD))
			return true;
		else
			return false;
	}

	
	@Override
	protected boolean isCachedSim() {
		// TODO Auto-generated method stub
		String measure = getMeasure();
		if (measure == null)
			return false;
		else if (measure.equals(PC))
			return false;
		else
			return super.isCachedSim();
	}


	@Override
	protected double sim0(String measure, RatingVector vRating1, RatingVector vRating2, Profile profile1, Profile profile2, Object...params) {
		// TODO Auto-generated method stub
		if (measure.equals(PSS))
			return pss(vRating1, vRating2, profile1, profile2);
		else if (measure.equals(NHSM))
			return nhsm(vRating1, vRating2, profile1, profile2);
		else if (measure.equals(BCF))
			return bcf(vRating1, vRating2, profile1, profile2);
		else if (measure.equals(BCFJ))
			return bcfj(vRating1, vRating2, profile1, profile2);
		else if (measure.equals(SRC))
			return src(vRating1, vRating2, profile1, profile2);
		else if (measure.equals(PIP))
			return pip(vRating1, vRating2, profile1, profile2);
		else if (measure.equals(PC)) {
			if ((params == null) || (params.length < 1) || !(params[0] instanceof Number))
				return Constants.UNUSED;
			else {
				int fixedColumnId = ((Number)(params[0])).intValue();
				return pc(vRating1, vRating2, profile1, profile2, fixedColumnId);
			}
		}
		else if (measure.equals(MMD))
			return mmd(vRating1, vRating2, profile1, profile2);
		else if (measure.equals(CJACMD))
			return cosine(vRating1, vRating2, profile1, profile2) + mmd(vRating1, vRating2, profile1, profile2) + jaccard(vRating1, vRating2, profile1, profile2);
		else if (measure.equals(FENG))
			return feng(vRating1, vRating2, profile1, profile2);
		else if (measure.equals(MU))
			return mu(vRating1, vRating2, profile1, profile2);
		else if (measure.equals(SMTP))
			return smtp(vRating1, vRating2, profile1, profile2);
		else if (measure.equals(AMER))
			return amer(vRating1, vRating2, profile1, profile2, this.itemIds);
		else if (measure.equals(AMER2))
			return amer2(vRating1, vRating2, profile1, profile2);
		else if (measure.equals(AMER2J))
			return amer2j(vRating1, vRating2, profile1, profile2);
		else if (measure.equals(QUASI_TFIDF))
			return quasiTfIdf(vRating1, vRating2, profile1, profile2);
		else if (measure.equals(QUASI_TFIDF_JACCARD))
			return quasiTfIdfJaccard(vRating1, vRating2, profile1, profile2);
		else if (measure.equals(TA))
			return triangleArea(vRating1, vRating2, profile1, profile2);
		else if (measure.equals(TAJ))
			return triangleAreaJaccard(vRating1, vRating2, profile1, profile2);
		else if (measure.equals(COCO))
			return coco(vRating1, vRating2, profile1, profile2);
		else if (measure.equals(NNMS))
			return mmns(vRating1, vRating2, profile1, profile2);
		else
			return super.sim0(measure, vRating1, vRating2, profile1, profile2, params);
	}

	
	/**
	 * Calculating the PSS measure between two pairs. PSS measure is developed by Haifeng Liu, Zheng Hu, Ahmad Mian, Hui Tian, Xuzhen Zhu, and implemented by Loc Nguyen.
	 * The first pair includes the first rating vector and the first profile.
	 * The second pair includes the second rating vector and the second profile.
	 * 
	 * @param vRating1 first rating vector.
	 * @param vRating2 second rating vector.
	 * @param profile1 first profile.
	 * @param profile2 second profile.
	 * @author Haifeng Liu, Zheng Hu, Ahmad Mian, Hui Tian, Xuzhen Zhu.
	 * @return PSS measure between both two rating vectors and profiles.
	 */
	protected abstract double pss(RatingVector vRating1, RatingVector vRating2,
			Profile profile1, Profile profile2);


	/**
	 * Calculating the PSS measure between two rating vectors. PSS measure is developed by Haifeng Liu, Zheng Hu, Ahmad Mian, Hui Tian, Xuzhen Zhu, and implemented by Loc Nguyen.
	 * @param vRating1 first rating vector.
	 * @param vRating2 second rating vector.
	 * @param fieldMeans map of field means.
	 * @author Haifeng Liu, Zheng Hu, Ahmad Mian, Hui Tian, Xuzhen Zhu.
	 * @return PSS measure between two rating vectors.
	 */
	protected double pss(RatingVector vRating1, RatingVector vRating2, Map<Integer, Double> fieldMeans) {
		Set<Integer> common = commonFieldIds(vRating1, vRating2);
		if (common.size() == 0) return Constants.UNUSED;
		
		double pss = 0.0;
		for (int id : common) {
			double r1 = vRating1.get(id).value;
			double r2 = vRating2.get(id).value;
			
			double pro = 1.0 - 1.0 / (1.0 + Math.exp(-Math.abs(r1-r2)));
			//Note: I think that it is better to use mean instead of median for significant.
			//At the worst case, median is always approximate to mean given symmetric distribution like normal distribution.
			//Moreover, in fact, general user mean is equal to general item mean.
			//However, I still use rating median because of respecting authors' ideas.
			double sig = 1.0 / (1.0 + Math.exp(
					-Math.abs(r1-ratingMedian)*Math.abs(r2-ratingMedian)));
			double singular = 1.0 - 1.0 / (1.0 + Math.exp(-Math.abs((r1+r2)/2.0 - fieldMeans.get(id))));
			
			pss += pro * sig * singular;
		}
		
		return pss;
	}
	
	
	/**
	 * Calculating the NHSM measure between two pairs. NHSM measure is developed by Haifeng Liu, Zheng Hu, Ahmad Mian, Hui Tian, Xuzhen Zhu, and implemented by Loc Nguyen.
	 * The first pair includes the first rating vector and the first profile.
	 * The second pair includes the second rating vector and the second profile.
	 * 
	 * @param vRating1 first rating vector.
	 * @param vRating2 second rating vector.
	 * @param profile1 first profile.
	 * @param profile2 second profile.
	 * @author Haifeng Liu, Zheng Hu, Ahmad Mian, Hui Tian, Xuzhen Zhu.
	 * @return NHSM measure between both two rating vectors and profiles.
	 */
	protected double nhsm(RatingVector vRating1, RatingVector vRating2,
			Profile profile1, Profile profile2) {
		double urp = urp(vRating1, vRating2, profile1, profile2);
		double jaccard2 = jaccard2(vRating1, vRating2, profile1, profile2);
		return pss(vRating1, vRating2, profile1, profile2) * jaccard2 * urp;
	}


	/**
	 * Calculate the Bhattacharyya measure from specified rating vectors. BC measure is modified by Bidyut Kr. Patra, Raimo Launonen, Ville Ollikainen, Sukumar Nandi, and implemented by Loc Nguyen.
	 * @param vRating1 first rating vector.
	 * @param vRating2 second rating vector.
	 * @param profile1 first profile.
	 * @param profile2 second profile.
	 * @author Bidyut Kr. Patra, Raimo Launonen, Ville Ollikainen, Sukumar Nandi.
	 * @return Bhattacharyya measure from specified rating vectors.
	 */
	@NextUpdate
	protected double bc(RatingVector vRating1, RatingVector vRating2,
			Profile profile1, Profile profile2) {
		Task task = new Task() {
			
			@Override
			public Object perform(Object...params) {
				List<Double> bins = valueBins;
				if (bins.isEmpty())
					bins = extractValueBins(vRating1, vRating2);
				
				Set<Integer> ids1 = vRating1.fieldIds(true);
				Set<Integer> ids2 = vRating2.fieldIds(true);
				int n1 = ids1.size();
				int n2 = ids2.size();
				if (n1 == 0 || n2 == 0) return Constants.UNUSED;
				
				double bc = 0;
				for (double bin : bins) {
					int count1 = 0, count2 = 0;
					for (int id1 : ids1) {
						if (vRating1.get(id1).value == bin)
							count1++;
					}
					for (int id2 : ids2) {
						if (vRating2.get(id2).value == bin)
							count2++;
					}
					
					bc += Math.sqrt( ((double)count1/(double)n1) * ((double)count2/(double)n2) ); 
				}
				
				return bc;
			}
		};
		
		return (double)cacheTask(vRating1.id(), vRating2.id(), this.columnSimCache, task);
	}

	
	/**
	 * Calculating the advanced BCF measure between two pairs. BCF measure is developed by Bidyut Kr. Patra, Raimo Launonen, Ville Ollikainen, Sukumar Nandi, and implemented by Loc Nguyen.
	 * The first pair includes the first rating vector and the first profile.
	 * The second pair includes the second rating vector and the second profile.
	 * @param vRating1 first rating vector.
	 * @param vRating2 second rating vector.
	 * @param profile1 first profile.
	 * @param profile2 second profile.
	 * @author Bidyut Kr. Patra, Raimo Launonen, Ville Ollikainen, Sukumar Nandi.
	 * @return BCF measure between both two rating vectors and profiles.
	 */
	@NextUpdate
	protected double bcf(RatingVector vRating1, RatingVector vRating2,
			Profile profile1, Profile profile2) {
		
		Set<Integer> columnIds1 = vRating1.fieldIds(true);
		Set<Integer> columnIds2 = vRating2.fieldIds(true);
		if (columnIds1.size() == 0 || columnIds2.size() == 0)
			return Constants.UNUSED;
		
		double bcSum = 0;
		boolean medianMode = getConfig().getAsBoolean(BCF_MEDIAN_MODE_FIELD);
		for (int columnId1 : columnIds1) {
			RatingVector columnVector1 = getColumnRating(columnId1);
			if (columnVector1 == null) continue;
			double columnModule1 = bcfCalcColumnModule(columnVector1);
			if (!Util.isUsed(columnModule1) || columnModule1 == 0) continue;
			
			double value1 = medianMode? vRating1.get(columnId1).value-this.ratingMedian : vRating1.get(columnId1).value-vRating1.mean();
			for (int columnId2 : columnIds2) {
				RatingVector columnVector2 = columnId2 == columnId1 ? columnVector1 : getColumnRating(columnId2);
				if (columnVector2 == null) continue;
				double columnModule2 = bcfCalcColumnModule(columnVector2);
				if (!Util.isUsed(columnModule2) || columnModule2 == 0) continue;
				
				double bc = bc(columnVector1, columnVector2, profile1, profile2);
				if (!Util.isUsed(bc)) continue;

				double value2 = medianMode? vRating2.get(columnId2).value-this.ratingMedian : vRating2.get(columnId2).value-vRating2.mean();
				double loc = value1 * value2 / (columnModule1*columnModule2);
				if (!Util.isUsed(loc)) continue;
				
				bcSum += bc * loc;
			}
		}
		
		return bcSum;
	}

	
	/**
	 * Calculating the advanced BCFJ measure (BCF + Jaccard) between two pairs. BCF measure is developed by Bidyut Kr. Patra, Raimo Launonen, Ville Ollikainen, Sukumar Nandi, and implemented by Loc Nguyen.
	 * The first pair includes the first rating vector and the first profile.
	 * The second pair includes the second rating vector and the second profile.
	 * @param vRating1 first rating vector.
	 * @param vRating2 second rating vector.
	 * @param profile1 first profile.
	 * @param profile2 second profile.
	 * @author Bidyut Kr. Patra, Raimo Launonen, Ville Ollikainen, Sukumar Nandi.
	 * @return BCFJ measure between both two rating vectors and profiles.
	 */
	protected double bcfj(RatingVector vRating1, RatingVector vRating2,
			Profile profile1, Profile profile2) {
		return bcf(vRating1, vRating2, profile1, profile2) + jaccard(vRating1, vRating2, profile1, profile2);
	}
	
	
	/**
	 * Calculating module (length) of column rating vector for BCF measure.
	 * @param columnVector specified column rating vector.
	 * @return module (length) of column rating vector.
	 */
	protected double bcfCalcColumnModule(RatingVector columnVector) {
		double ratingMedian = this.ratingMedian;
		Task task = new Task() {
			
			@Override
			public Object perform(Object...params) {
				if (columnVector == null) return Constants.UNUSED;
				
				Set<Integer> fieldIds = columnVector.fieldIds(true);
				double columnModule = 0;
				boolean medianMode = getConfig().getAsBoolean(BCF_MEDIAN_MODE_FIELD);
				for (int fieldId : fieldIds) {
					double deviate = medianMode ? columnVector.get(fieldId).value-ratingMedian : columnVector.get(fieldId).value;
					columnModule += deviate * deviate;
				}
				
				return Math.sqrt(columnModule);
			}
		};
		
		return (double)cacheTask(columnVector.id(), this.bcfColumnModuleCache, task);
	}

	
	/**
	 * Calculating the Spearman Rank Correlation (SRC) measure between two pairs.
	 * The first pair includes the first rating vector and the first profile.
	 * The second pair includes the second rating vector and the second profile.
	 * 
	 * @param vRating1 first rating vector.
	 * @param vRating2 second rating vector.
	 * @param profile1 first profile.
	 * @param profile2 second profile.
	 * @return Spearman Rank Correlation (SRC) measure between both two rating vectors and profiles.
	 */
	protected double src(RatingVector vRating1, RatingVector vRating2,
			Profile profile1, Profile profile2) {
		Map<Double, Integer> bins = rankBins;
		if (bins.isEmpty())
			bins = extractRankBins(vRating1, vRating2);

		Set<Integer> common = commonFieldIds(vRating1, vRating2);
		if (common.size() == 0) return Constants.UNUSED;
		
		double sum = 0;
		for (int id : common) {
			double v1 = vRating1.get(id).value;
			int r1 = bins.get(v1);
			double v2 = vRating2.get(id).value;
			int r2 = bins.get(v2);
			
			int d = r1 - r2;
			sum += d*d;
		}
		
		double n = common.size();
		return 1.0 - 6*sum/(n*(n*n-1));
	}
	
	
	/**
	 * Calculating the PIP measure between two pairs. PIP measure is developed by Hyung Jun Ahn, and implemented by Loc Nguyen.
	 * The first pair includes the first rating vector and the first profile.
	 * The second pair includes the second rating vector and the second profile.
	 * 
	 * @param vRating1 first rating vector.
	 * @param vRating2 second rating vector.
	 * @param profile1 first profile.
	 * @param profile2 second profile.
	 * @author Hyung Jun Ahn.
	 * @return NHSM measure between both two rating vectors and profiles.
	 */
	protected abstract double pip(RatingVector vRating1, RatingVector vRating2,
			Profile profile1, Profile profile2);
	
	
	/**
	 * Calculating the PIP measure between two rating vectors. PIP measure is developed by Hyung Jun Ahn and implemented by Loc Nguyen.
	 * @param vRating1 first rating vector.
	 * @param vRating2 second rating vector.
	 * @param fieldMeans map of field means.
	 * @author Hyung Jun Ahn
	 * @return PIP measure between two rating vectors.
	 */
	protected double pip(RatingVector vRating1, RatingVector vRating2, Map<Integer, Double> fieldMeans) {
		Set<Integer> common = commonFieldIds(vRating1, vRating2);
		if (common.size() == 0) return Constants.UNUSED;
		
		double pip = 0.0;
		for (int id : common) {
			double r1 = vRating1.get(id).value;
			double r2 = vRating2.get(id).value;
			boolean agreed = agree(r1, r2);
			
			double d = agreed ? Math.abs(r1-r2) : 2*Math.abs(r1-r2);
			double pro = (2*(config.getMaxRating()-config.getMinRating())+1) - d;
			pro = pro*pro;
			
			double impact = (Math.abs(r1-ratingMedian)+1) * (Math.abs(r2-ratingMedian)+1);
			if (!agreed)
				impact = 1 / impact;
			
			double mean = fieldMeans.get(id);
			double pop = 1;
			if ((r1 > mean && r2 > mean) || (r1 < mean && r2 < mean)) {
				double bias = (r1+r2)/2 - mean;
				pop = 1 + bias*bias;
			}
			
			pip += pro * impact * pop;
		}
		
		return pip;
	}

	
	/**
	 * Checking whether two ratings are agreed.
	 * @param rating1 first rating.
	 * @param rating2 second rating.
	 * @return true if two ratings are agreed.
	 */
	protected boolean agree(double rating1, double rating2) {
		if ( (rating1 > this.ratingMedian && rating2 < this.ratingMedian) || (rating1 < this.ratingMedian && rating2 > this.ratingMedian) )
			return false;
		else
			return true;
	}
	
	
	/**
	 * Calculating the PC measure between two rating vectors. PC measure is developed by Keunho Choi and Yongmoo Suh. It implemented by Loc Nguyen.
	 * The first pair includes the first rating vector and the first profile.
	 * The second pair includes the second rating vector and the second profile.
	 * 
	 * @param vRating1 first rating vector.
	 * @param vRating2 second rating vector.
	 * @param profile1 first profile.
	 * @param profile2 second profile.
	 * @param fixedColumnId fixed column identifier.
	 * @author Hyung Jun Ahn.
	 * @return PC measure between both two rating vectors and profiles.
	 */
	protected abstract double pc(RatingVector vRating1, RatingVector vRating2,
			Profile profile1, Profile profile2, int fixedColumnId);
	
	
	/**
	 * Calculating the PC measure between two rating vectors. PC measure is developed by Keunho Choi and Yongmoo Suh. It implemented by Loc Nguyen.
	 * @param vRating1 the first rating vectors.
	 * @param vRating2 the second rating vectors.
	 * @param fixedColumnId fixed field (column) identifier.
	 * @param fieldMeans mean value of field ratings.
	 * @author Keunho Choi, Yongmoo Suh
	 * @return PC measure between two rating vectors.
	 */
	protected double pc(RatingVector vRating1, RatingVector vRating2, int fixedColumnId, Map<Integer, Double> fieldMeans) {
		Set<Integer> common = commonFieldIds(vRating1, vRating2);
		if (common.size() == 0) return Constants.UNUSED;

		double vx = 0, vy = 0;
		double vxy = 0;
		for (int fieldId : common) {
			double mean = fieldMeans.get(fieldId);
			double d1 = vRating1.get(fieldId).value - mean;
			double d2 = vRating2.get(fieldId).value - mean;
			
			Task columnSimTask = new Task() {
				
				@Override
				public Object perform(Object...params) {
					RatingVector fixedColumnVector = getColumnRating(fixedColumnId);
					RatingVector columnVector = getColumnRating(fieldId);
					
					if (fixedColumnVector == null || columnVector == null)
						return Constants.UNUSED;
					else
						return fixedColumnVector.corr(columnVector);
				}
			};
			double columnSim = (double)cacheTask(fixedColumnId, fieldId, this.columnSimCache, columnSimTask);
			columnSim = columnSim * columnSim;
			
			vx  += d1 * d1 * columnSim;
			vy  += d2 * d2 * columnSim;
			vxy += d1 * d2 * columnSim;
		}
		
		if (vx == 0 || vy == 0)
			return Constants.UNUSED;
		else
			return vxy / Math.sqrt(vx * vy);
	}

	
	/**
	 * Calculating the Mean Measure of Divergence (MMD) measure between two pairs.
	 * Suryakant and Tripti Mahara proposed use of MMD for collaborative filtering. Loc Nguyen implements it.
	 * The first pair includes the first rating vector and the first profile.
	 * The second pair includes the second rating vector and the second profile.
	 * @param vRating1 first rating vector.
	 * @param vRating2 second rating vector.
	 * @param profile1 first profile.
	 * @param profile2 second profile.
	 * @author Suryakant, Tripti Mahara
	 * @return MMD measure between both two rating vectors and profiles.
	 */
	protected double mmd(RatingVector vRating1, RatingVector vRating2,
			Profile profile1, Profile profile2) {
		Set<Integer> ids1 = vRating1.fieldIds(true);
		Set<Integer> ids2 = vRating2.fieldIds(true);
		int N1 = ids1.size();
		int N2 = ids2.size();
		if (N1 == 0 || N2 == 0) return Constants.UNUSED;
		
		List<Double> bins = valueBins;
		if (bins.isEmpty())
			bins = extractValueBins(vRating1, vRating2);
		double sum = 0;
		for (double bin : bins) {
			int n1 = 0, n2 = 0;
			for (int id1 : ids1) {
				if (vRating1.get(id1).value == bin)
					n1++;
			}
			for (int id2 : ids2) {
				if (vRating2.get(id2).value == bin)
					n2++;
			}
			
			double thetaBias = mmdTheta(n1, N1) - mmdTheta(n2, N2);
			sum += thetaBias*thetaBias - 1/(0.5+n1) - 1/(0.5+n2); 
		}
		
		return 1 / (1 + sum/bins.size());
	}
	
	
	/**
	 * Theta transformation of Mean Measure of Divergence (MMD) measure.
	 * The default implementation is Grewal transformation.
	 * @param n number of observations having a trait.
	 * @param N number of observations
	 * @return Theta transformation of Mean Measure of Devergence (MMD) measure.
	 */
	protected double mmdTheta(int n, int N) {
		return 1 / Math.sin(1-2*(n/N));
	}

	
	/**
	 * Calculating the Feng measure between two pairs.
	 * Junmei Feng, Xiaoyi Fengs, Ning Zhang, and Jinye Peng developed the Triangle measure. Loc Nguyen implements it.
	 * The first pair includes the first rating vector and the first profile.
	 * The second pair includes the second rating vector and the second profile.
	 * @param vRating1 first rating vector.
	 * @param vRating2 second rating vector.
	 * @param profile1 first profile.
	 * @param profile2 second profile.
	 * @author Junmei Feng, Xiaoyi Fengs, Ning Zhang, Jinye Peng
	 * @return Feng measure between both two rating vectors and profiles.
	 */
	protected double feng(RatingVector vRating1, RatingVector vRating2,
			Profile profile1, Profile profile2) {
		
		double s1 = coj(vRating1, vRating2, profile1, profile2);

		Set<Integer> ids1 = vRating1.fieldIds(true);
		Set<Integer> ids2 = vRating2.fieldIds(true);
		Set<Integer> common = Util.newSet();
		common.addAll(ids1);
		common.retainAll(ids2);
		double s2 = 1 / ( 1 + Math.exp(-common.size()*common.size()/(ids1.size()*ids2.size())) );
		
		double s3 = urp(vRating1, vRating2, profile1, profile2);
		
		return s1 * s2 * s3;
	}
	
	
	/**
	 * Calculating the Mu measure between two pairs.
	 * Yi Mua, Nianhao Xiao, Ruichun Tang, Liang Luo, and Xiaohan Yin developed Mu measure. Loc Nguyen implements it.
	 * The first pair includes the first rating vector and the first profile.
	 * The second pair includes the second rating vector and the second profile.
	 * @param vRating1 first rating vector.
	 * @param vRating2 second rating vector.
	 * @param profile1 first profile.
	 * @param profile2 second profile.
	 * @author Yi Mua, Nianhao Xiao, Ruichun Tang, Liang Luo, Xiaohan Yin
	 * @return Mu measure between both two rating vectors and profiles.
	 */
	@NextUpdate
	protected double mu(RatingVector vRating1, RatingVector vRating2,
			Profile profile1, Profile profile2) {
		double alpha = config.getAsReal(MU_ALPHA_FIELD);
		double pearson = corr(vRating1, vRating2, profile1, profile2);
		double hg = 1 - bc(vRating1, vRating2, profile1, profile2);
//		double hg = bc(vRating1, vRating2, profile1, profile2);
		double jaccard = jaccard(vRating1, vRating2, profile1, profile2);
		
		return alpha*pearson + (1-alpha)*(hg+jaccard);
	}
	
	
	/**
	 * Calculating the SMTP measure between two pairs. SMTP is developed by Yung-Shen Lin, Jung-Yi Jiang, Shie-Jue Lee, and implemented by Loc Nguyen.
	 * The first pair includes the first rating vector and the first profile.
	 * The second pair includes the second rating vector and the second profile.
	 * 
	 * @param vRating1 first rating vector.
	 * @param vRating2 second rating vector.
	 * @param profile1 first profile.
	 * @param profile2 second profile.
	 * @author Yung-Shen Lin, Jung-Yi Jiang, Shie-Jue Lee.
	 * @return SMTP measure between both two rating vectors.
	 */
	protected double smtp(
			RatingVector vRating1, RatingVector vRating2,
			Profile profile1, Profile profile2) {
		
		List<Integer> common = commonFieldIdsAsList(vRating1, vRating2);
		common.retainAll(this.itemVars.keySet());
		if (common.size() == 0) return Constants.UNUSED;
		
		double[] data1 = new double[common.size()];
		double[] data2 = new double[common.size()];
		double[] vars = new double[common.size()];
		boolean useGeneralVar = getConfig().getAsBoolean(SMTP_GENERAL_VAR_FIELD);
		for (int i = 0; i < common.size(); i++) {
			int id = common.get(i);
			
			data1[i] = vRating1.get(id).value; 
			data2[i] = vRating2.get(id).value;
			if (useGeneralVar)
				vars[i] = this.ratingVar;
			else
				vars[i] = this.itemVars.get(id);
		}

		DocumentVector vector1 = new DocumentVector(data1);
		DocumentVector vector2 = new DocumentVector(data2);
		
		double lamda = getConfig().getAsReal(SMTP_LAMBDA_FIELD);
		return vector1.smtp(vector2, lamda, vars);
	}
	
	
	/**
	 * Calculating the Amer measure between two pairs. Amer measure is developed by Ali Amer, and implemented by Loc Nguyen.
	 * The first pair includes the first rating vector and the first profile.
	 * The second pair includes the second rating vector and the second profile.
	 * 
	 * @param vRating1 first rating vector.
	 * @param vRating2 second rating vector.
	 * @param profile1 first profile.
	 * @param profile2 second profile.
	 * @param itemIds set of all item identifiers
	 * @author Ali Amer.
	 * @return Amer measure between both two rating vectors and profiles.
	 */
	protected double amer(
			RatingVector vRating1, RatingVector vRating2,
			Profile profile1, Profile profile2, Set<Integer> itemIds) {
		if (itemIds == null)
			itemIds = Util.newSet();
		itemIds.addAll(unionFieldIds(vRating1, vRating2));
		int N = itemIds.size();
		if (N == 0) return Constants.UNUSED;
		
		int Na = 0, Nb = 0, Nab = 0, F = 0;
		for (int itemId : itemIds) {
			boolean rated1 = vRating1.isRated(itemId);
			boolean rated2 = vRating2.isRated(itemId);
			
			if (rated1) Na++;
			if (rated2) Nb++;
			if (rated1 && rated2) Nab++;
			if ((rated1 && !rated2) || (!rated1 && rated2)) F++;
		}
		
		return ((1.0 - F/N) + (2.0*Nab / (Na + Nb))) / 2.0;
	}
	
	
	/**
	 * Calculating the Amer2 measure between two pairs. Amer2 measure is developed by Ali Amer, and implemented by Loc Nguyen.
	 * Amer2 measure is only applied into positive ratings.
	 * The first pair includes the first rating vector and the first profile.
	 * The second pair includes the second rating vector and the second profile.
	 * 
	 * @param vRating1 first rating vector.
	 * @param vRating2 second rating vector.
	 * @param profile1 first profile.
	 * @param profile2 second profile.
	 * @author Ali Amer.
	 * @return Amer2 measure between both two rating vectors and profiles.
	 */
	protected double amer2(
			RatingVector vRating1, RatingVector vRating2,
			Profile profile1, Profile profile2) {
		Set<Integer> itemIds = unionFieldIds(vRating1, vRating2);
		if (itemIds.size() == 0) return Constants.UNUSED;
		
		double X = 0, Y = 0, U = 0, V = 0;
		for (int itemId : itemIds) {
			boolean rated1 = vRating1.isRated(itemId);
			boolean rated2 = vRating2.isRated(itemId);
			
			if (rated1) {
				double value = vRating1.get(itemId).value;
				U += value;
				if (!rated2) X += value;
			}
			if (rated2) {
				double value = vRating2.get(itemId).value;
				V += value;
				if (!rated1) Y += value;
			}
		}
		
		double F = X * Y;
		double N = U * V;
		return 1.0 - (F + 1.0) / N;
	}

	
	/**
	 * Calculating the Amer2 + Jaccard measure between two pairs. Amer2 measure is developed by Ali Amer, and implemented by Loc Nguyen.
	 * Amer2 + Jaccard measure is only applied into positive ratings.
	 * The first pair includes the first rating vector and the first profile.
	 * The second pair includes the second rating vector and the second profile.
	 * 
	 * @param vRating1 first rating vector.
	 * @param vRating2 second rating vector.
	 * @param profile1 first profile.
	 * @param profile2 second profile.
	 * @author Ali Amer.
	 * @return Amer2 + Jaccard measure between both two rating vectors and profiles.
	 */
	protected double amer2j(
			RatingVector vRating1, RatingVector vRating2,
			Profile profile1, Profile profile2) {
		return amer2(vRating1, vRating2, profile1, profile2) * jaccard(vRating1, vRating2, profile1, profile2);
	}
	
	
	/**
	 * Calculating the quasi-TfIdf measure between two pairs. Quasi-TfIdf measure is developed by Ali Amer and Loc Nguyen.
	 * Quasi-TfIdf measure is an extension of Amer2 measure and the ideology of TF and IDF.
	 * Quasi-TfIdf measure is only applied into positive ratings.
	 * The first pair includes the first rating vector and the first profile.
	 * The second pair includes the second rating vector and the second profile.
	 * 
	 * @param vRating1 first rating vector.
	 * @param vRating2 second rating vector.
	 * @param profile1 first profile.
	 * @param profile2 second profile.
	 * @author Ali Amer, Loc Nguyen.
	 * @return Quasi-TfIdf measure between both two rating vectors.
	 */
	protected double quasiTfIdf(
			RatingVector vRating1, RatingVector vRating2,
			Profile profile1, Profile profile2) {
		Set<Integer> itemIds = unionFieldIds(vRating1, vRating2);
		if (itemIds.size() == 0) return Constants.UNUSED;
		
		double X1 = 0, Y1 = 0, X2 = 0, Y2 = 0, U = 0, V = 0;
		for (int itemId : itemIds) {
			boolean rated1 = vRating1.isRated(itemId);
			boolean rated2 = vRating2.isRated(itemId);
			
			if (rated1) {
				double value1 = vRating1.get(itemId).value;
				U += value1;
				if (rated2)
					X1 += value1;
				else if (!rated2)
					X2 += value1;
			}
			
			if (rated2) {
				double value2 = vRating2.get(itemId).value;
				V += value2;
				if (rated1)
					Y1 += value2;
				else if (!rated1)
					Y2 += value2;
			}
		}
		
		double N = U * V;
		return ((X1*Y1)/N) * (1.0 - (X2*Y2)/N);
	}

	
	/**
	 * Calculating the quasi-TfIdf + Jaccard measure between two pairs. Quasi-TfIdf measure is developed by Ali Amer and Loc Nguyen.
	 * Quasi-TfIdf + Jaccard measure is an extension of Amer2 measure and the ideology of TF and IDF.
	 * Quasi-TfIdf + Jaccard measure is only applied into positive ratings.
	 * The first pair includes the first rating vector and the first profile.
	 * The second pair includes the second rating vector and the second profile.
	 * 
	 * @param vRating1 first rating vector.
	 * @param vRating2 second rating vector.
	 * @param profile1 first profile.
	 * @param profile2 second profile.
	 * @author Ali Amer, Loc Nguyen.
	 * @return Quasi-TfIdf + Jaccard measure between both two rating vectors.
	 */
	protected double quasiTfIdfJaccard(
			RatingVector vRating1, RatingVector vRating2,
			Profile profile1, Profile profile2) {
		Set<Integer> itemIds = unionFieldIds(vRating1, vRating2);
		if (itemIds.size() == 0) return Constants.UNUSED;
		
		double X1 = 0, Y1 = 0, X2 = 0, Y2 = 0, U = 0, V = 0;
		int commonCount = 0;
		for (int itemId : itemIds) {
			boolean rated1 = vRating1.isRated(itemId);
			boolean rated2 = vRating2.isRated(itemId);
			
			if (rated1) {
				double value1 = vRating1.get(itemId).value;
				U += value1;
				if (rated2) {
					X1 += value1;
					commonCount ++;
				}
				else if (!rated2)
					X2 += value1;
			}
			
			if (rated2) {
				double value2 = vRating2.get(itemId).value;
				V += value2;
				if (rated1)
					Y1 += value2;
				else if (!rated1)
					Y2 += value2;
			}
		}
		
		double N = U * V;
		double jac = (double)commonCount / (double)itemIds.size();
		return ((X1*Y1)*jac/N) * (1.0 - (X2*Y2)*(1.0-jac)/N);
	}

	
	/**
	 * Calculating the TA (triangle area) measure between two pairs. TA is developed by Loc Nguyen.
	 * The first pair includes the first rating vector and the first profile.
	 * The second pair includes the second rating vector and the second profile.
	 * The current version does not support positive cosine. The next version will fix it.
	 * 
	 * @param vRating1 first rating vector.
	 * @param vRating2 second rating vector.
	 * @param profile1 first profile.
	 * @param profile2 second profile.
	 * @author Loc Nguyen.
	 * @return TA measure between both two rating vectors and profiles.
	 */
	protected double triangleArea(RatingVector vRating1, RatingVector vRating2,
			Profile profile1, Profile profile2) {
		Set<Integer> common = commonFieldIds(vRating1, vRating2);
		if (common.size() == 0) return Constants.UNUSED;
		
		Vector2 v1 = new Vector2(common.size(), 0);
		Vector2 v2 = new Vector2(common.size(), 0);
		boolean normalized = getConfig().getAsBoolean(TA_NORMALIZED_FIELD);
		if (normalized) {//Normalized mode
			for (int id : common) {
				v1.add(vRating1.get(id).value - this.ratingMedian);
				v2.add(vRating2.get(id).value - this.ratingMedian);
			}
		}
		else {
			for (int id : common) {
				v1.add(vRating1.get(id).value);
				v2.add(vRating2.get(id).value);
			}
		}
		
		double a = v1.module();
		double b = v2.module();
		if (a == 0 || b == 0) return Constants.UNUSED;
		
		double p = v1.product(v2);
		if (p >= 0) {
			if (a < b)
				return p*p / (a*b*b*b);
			else
				return p*p / (a*a*a*b);
		}
		else {
			if (a < b)
				return p / (b*b);
			else
				return p / (a*a);
		}
	}

	
	/**
	 * Calculating the TAJ (triangle area + Jaccard) measure between two pairs. TAJ is developed by Loc Nguyen.
	 * The first pair includes the first rating vector and the first profile.
	 * The second pair includes the second rating vector and the second profile.
	 * The current version does not support positive cosine. The next version will fix it.
	 * 
	 * @param vRating1 first rating vector.
	 * @param vRating2 second rating vector.
	 * @param profile1 first profile.
	 * @param profile2 second profile.
	 * @author Loc Nguyen
	 * @return TAJ measure between both two rating vectors and profiles.
	 */
	protected double triangleAreaJaccard(RatingVector vRating1, RatingVector vRating2,
			Profile profile1, Profile profile2) {
		return triangleArea(vRating1, vRating2, profile1, profile2) * jaccard(vRating1, vRating2, profile1, profile2);
	}

	
	/**
	 * Calculating the circle dot product of two rating vectors given two sets of field identifiers.
	 * @param vRating1 rating vector 1.
	 * @param vRating2 rating vector 1.
	 * @param fieldIds1 set 1 of field identifiers.
	 * @param fieldIds2 set 1 of field identifiers.
	 * @return circle dot product of two rating vectors given two sets of field identifiers.
	 */
	private double circleDotProduct0(RatingVector vRating1, RatingVector vRating2, Set<Integer> fieldIds1, Set<Integer> fieldIds2) {
		double product = 0;
		for (int fieldId1 : fieldIds1) {
			double value1 = vRating1.get(fieldId1).value;
			
			for (int fieldId2 : fieldIds2) {
				double value2 = vRating2.get(fieldId2).value;
				product += value1 * value2;
			}
		}
		
		return product;
	}
	
	
	/**
	 * Calculating the circle dot product of two rating vectors given two sets of field identifiers.
	 * @param vRating1 rating vector 1.
	 * @param vRating2 rating vector 1.
	 * @param fieldIds1 set 1 of field identifiers.
	 * @param fieldIds2 set 1 of field identifiers.
	 * @return circle dot product of two rating vectors given two sets of field identifiers.
	 */
	protected double circleDotProduct(RatingVector vRating1, RatingVector vRating2, Set<Integer> fieldIds1, Set<Integer> fieldIds2) {
		double product = 0;
		for (int fieldId1 : fieldIds1) {
			if (!vRating1.isRated(fieldId1)) continue;
			
			double value1 = vRating1.get(fieldId1).value;
			for (int fieldId2 : fieldIds2) {
				if (!vRating2.isRated(fieldId2)) continue;

				double value2 = vRating2.get(fieldId2).value;
				product += value1 * value2;
			}
		}
		
		return product;
	}

	
	/**
	 * Calculating the circle dot product of two rating vectors.
	 * @param vRating1 rating vector 1.
	 * @param vRating2 rating vector 1.
	 * @return circle dot product of two rating vectors.
	 */
	protected double circleDotProduct(RatingVector vRating1, RatingVector vRating2) {
		return circleDotProduct0(vRating1, vRating2,
				vRating1.fieldIds(true), vRating2.fieldIds(true));
	}

	
	/**
	 * Calculating the circle dot length of specified rating vector given field identifiers.
	 * @param vRating specified rating vector.
	 * @param fieldIds given field identifiers.
	 * @return circle dot length of specified rating vector given field identifiers.
	 */
	private double circleLength0(RatingVector vRating, Set<Integer> fieldIds) {
		double length = 0;
		for (int fieldId : fieldIds) {
			double value = vRating.get(fieldId).value;
			length += value*value;
		}
		
		return Math.sqrt(length);
	}

	
	/**
	 * Calculating the circle length of specified rating vector given field identifiers.
	 * @param vRating specified rating vector.
	 * @param fieldIds given field identifiers.
	 * @return circle length of specified rating vector given field identifiers.
	 */
	protected double circleLength(RatingVector vRating, Set<Integer> fieldIds) {
		double length = 0;
		for (int fieldId : fieldIds) {
			if (!vRating.isRated(fieldId)) continue;
			
			double value = vRating.get(fieldId).value;
			length += value*value;
		}
		
		return Math.sqrt(length);
	}

	
	/**
	 * Calculating the circle length of specified rating vector.
	 * @param vRating specified rating vector.
	 * @return circle length of specified rating vector.
	 */
	protected double circleLength(RatingVector vRating) {
		return circleLength0(vRating, vRating.fieldIds(true));
	}

	
	/**
	 * Calculating the coco measure of two rating vectors given two sets of field identifiers.
	 * @param vRating1 rating vector 1.
	 * @param vRating2 rating vector 1.
	 * @param fieldIds1 set 1 of field identifiers.
	 * @param fieldIds2 set 1 of field identifiers.
	 * @return coco measure of two rating vectors given two sets of field identifiers.
	 */
	protected double coco(RatingVector vRating1, RatingVector vRating2, Set<Integer> fieldIds1, Set<Integer> fieldIds2) {
		return circleDotProduct(vRating1, vRating2, fieldIds1, fieldIds2) /
				(circleLength(vRating1, fieldIds1)*circleLength(vRating2, fieldIds2));
	}

	
	/**
	 * Calculating the Coco measure between two pairs. Coco is developed by Loc Nguyen.
	 * The first pair includes the first rating vector and the first profile.
	 * The second pair includes the second rating vector and the second profile.
	 * 
	 * @param vRating1 first rating vector.
	 * @param vRating2 second rating vector.
	 * @param profile1 first profile.
	 * @param profile2 second profile.
	 * @author Loc Nguyen
	 * @return Coco measure between both two rating vectors and profiles.
	 */
	protected double coco(RatingVector vRating1, RatingVector vRating2,
			Profile profile1, Profile profile2) {
		Set<Integer> fieldIds1 = vRating1.fieldIds(true);
		double length1 = 0;
		for (int fieldId1 : fieldIds1) {
			double value1 = vRating1.get(fieldId1).value;
			length1 += value1*value1;
		}
		
		Set<Integer> fieldIds2 = vRating2.fieldIds(true);
		double length2 = 0;
		for (int fieldId2 : fieldIds2) {
			double value2 = vRating2.get(fieldId2).value;
			length2 += value2*value2;
		}
		
		return (vRating1.sum()*vRating2.sum()) / Math.sqrt(length1*length2);
	}

	
	/**
	 * Calculating the numerical nearby similarity measure (MMNS) between two pairs. MMNS is developed by Ali Amer.
	 * @param vRating1 first rating vector.
	 * @param vRating2 second rating vector.
	 * @param profile1 first profile.
	 * @param profile2 second profile.
	 * @author Ali Amer
	 * @return numerical nearby similarity measure (MMNS) between both two rating vectors and profiles.
	 */
	protected double mmns(
			RatingVector vRating1, RatingVector vRating2,
			Profile profile1, Profile profile2) {
		
		Set<Integer> fieldIds = commonFieldIds(vRating1, vRating2);
		int n = fieldIds.size();
		double product = 0;
		for (int fieldId : fieldIds) {
			double value1 = vRating1.get(fieldId).value;
			double value2 = vRating2.get(fieldId).value;
			product += value1 * value2;
		}
		
		double sum1 = vRating1.sum();
		int n1 = vRating1.count(true);
		double sum2 = vRating2.sum();
		int n2 = vRating2.count(true);
		
		return (n*product) / (n1*sum1+n2*sum2);
	}
	
	
	/**
	 * Getting rating vector given column ID (item ID or user ID) for BCF measure.
	 * @param columnId specified column ID (item ID or user ID).
	 * @return rating vector given column ID (item ID or user ID).
	 */
	protected abstract RatingVector getColumnRating(int columnId);

	
//	/**
//	 * Calculating the ITA (inverse triangle area) measure between two pairs. ITA is developed by Loc Nguyen.
//	 * The first pair includes the first rating vector and the first profile.
//	 * The second pair includes the second rating vector and the second profile.
//	 * 
//	 * @param vRating1 first rating vector.
//	 * @param vRating2 second rating vector.
//	 * @param profile1 first profile.
//	 * @param profile2 second profile.
//	 * @author Loc Nguyen.
//	 * @return ITA measure between both two rating vectors and profiles.
//	 */
//	@Deprecated
//	protected double inverseTriangleArea(RatingVector vRating1, RatingVector vRating2,
//			Profile profile1, Profile profile2) {
//		
//		Set<Integer> common = commonFieldIds(vRating1, vRating2);
//		if (common.size() == 0) return Constants.UNUSED;
//		
//		Vector2 v1 = new Vector2(common.size(), 0);
//		Vector2 v2 = new Vector2(common.size(), 0);
//		for (int id : common) {
//			v1.add(vRating1.get(id).value / 5.0); //Fix here
//			v2.add(vRating2.get(id).value / 5.0); //Fix here
//		}
//
//		double cos = v1.cosine(v2);
//		double sin = Math.sqrt(1 - cos*cos); //Fix here
//		return 1 - v1.module() * v2.module() * sin;
//		
////		double cos2 = Math.abs(cos);
////		double c = a > b ? b/a : a/b;
////		double sin = Math.sqrt(1 - cos2*cos2);
////		double area = Math.abs(cos2 - c) * sin;
////		return (1 - area) * cos;
//		
////		double c = a > b ? b/a : a/b;
////		double area = Math.abs(cos2 - c);
////		if (area < 0) {
////			int i = 0;
////			i = 1;
////		}
////		return (1 - area) * cos;
//		
////		double c = a > b ? b/a : a/b;
////		return c*cos*cos;
//		
////		double acos = a*cos;
////		double bcos = b*cos;
////		if (acos == b || bcos == a)
////			return cos;
////		else if (acos < b)
////			return (acos/b)*cos;
////		else
////			return (bcos/a)*cos;
//		
////		if (taMode == TAMODE_MINPROJECT) {
////			double acos = a*cos;
////			double bcos = b*cos;
////			if (acos <= b && bcos <= a) {
////				double acoeff = acos/b;
////				double bcoeff = bcos/a;
////				if (acoeff < bcoeff)
////					coeff = acoeff;
////				else
////					coeff = bcoeff; 
////			}
////			else if (acos < b)
////				coeff = acos/b;
////			else
////				coeff = bcos/a;
////		}
////		else if (taMode == TAMODE_MAXPROJECT) {
////			double acos = a*cos;
////			double bcos = b*cos;
////			if (acos == b || bcos == a)
////				coeff = 1;
////			else if (acos < b && bcos < a) {
////				double acoeff = acos/b;
////				double bcoeff = bcos/a;
////				if (acoeff < bcoeff)
////					coeff = bcoeff;
////				else
////					coeff = acoeff; 
////			}
////			else if (acos < b)
////				coeff = acos/b;
////			else
////				coeff = bcos/a;
////		}
////		else if (taMode == TAMODE_EQUALRADIUS) {
////			coeff = a < b ? a / b : b / a;
////		}
//	}
	

	/**
	 * Computing common field IDs of two rating vectors as list.
	 * @param vRating1 first rating vector.
	 * @param vRating2 second rating vector.
	 * @return common field IDs of two rating vectors.
	 */
	private static List<Integer> commonFieldIdsAsList(RatingVector vRating1, RatingVector vRating2) {
		List<Integer> common = Util.newList();
		common.addAll(vRating1.fieldIds(true));
		common.retainAll(vRating2.fieldIds(true));
		return common;
	}

	
	/**
	 * Converting value bins into rank bins.
	 * @param valueBins value bins
	 * @return rank bins.
	 */
	protected static Map<Double, Integer> convertValueBinsToRankBins(List<Double> valueBins) {
		if (valueBins == null || valueBins.size() == 0)
			return Util.newMap();
		
		Collections.sort(valueBins);
		Map<Double, Integer> rankBins = Util.newMap();
		int n = valueBins.size();
		for (int i = 0; i < n; i++) {
			rankBins.put(valueBins.get(i), n-i);
		}
		
		return rankBins;
	}
	
	
	/**
	 * Extracting value bins from two specified rating vectors.
	 * @param vRating1 first rating vector.
	 * @param vRating2 second rating vector.
	 * @return Extracted value bins from two specified rating vectors.
	 */
	protected static List<Double> extractValueBins(RatingVector vRating1, RatingVector vRating2) {
		Set<Double> values = Util.newSet();
		
		Set<Integer> ids1 = vRating1.fieldIds(true);
		for (int id1 : ids1) {
			double value1 = vRating1.get(id1).value;
			values.add(value1);
		}
		
		Set<Integer> ids2 = vRating2.fieldIds(true);
		for (int id2 : ids2) {
			double value2 = vRating2.get(id2).value;
			values.add(value2);
		}
		
		List<Double> bins = DSUtil.toDoubleList(values);
		Collections.sort(bins);
		return bins;
	}
	
	
	/**
	 * Extracting rank bins from two specified rating vectors.
	 * @param vRating1 first rating vector.
	 * @param vRating2 second rating vector.
	 * @return Extracted rank bins from two specified rating vectors.
	 */
	protected static Map<Double, Integer> extractRankBins(RatingVector vRating1, RatingVector vRating2) {
		List<Double> valueBins = extractValueBins(vRating1, vRating2);
		return convertValueBinsToRankBins(valueBins);
	}
	
	
	/**
	 * Extracting value bins from configuration.
	 * @return extracted value bins from configuration.
	 */
	protected List<Double> extractConfigValueBins() {
		if (!getConfig().containsKey(VALUE_BINS_FIELD))
			return Util.newList();
		
		return TextParserUtil.parseListByClass(
				getConfig().getAsString(VALUE_BINS_FIELD),
				Double.class,
				",");
	}
	
	
	/**
	 * Extracting rank bins from configuration.
	 * @return extracted SRC rank bins from configuration.
	 */
	protected Map<Double, Integer> extractConfigRankBins() {
		List<Double> valueBins = extractConfigValueBins();
		return convertValueBinsToRankBins(valueBins);
	}

	
//	/**
//	 * Computing similarity matrix.
//	 * @param cf referred neighbor collaborative filtering.
//	 * @param vFetcher fetcher of ratings.
//	 * @return similarity matrix.
//	 */
//	protected static Map<Integer, Map<Integer, Double>> computeSimMatrix(NeighborCF cf, Fetcher<RatingVector> vFetcher) {
//		if (vFetcher == null) return Util.newMap();
//		int size = -1;
//		try {
//			size = vFetcher.getMetadata().getSize();
//		}
//		catch (Throwable e) {
//			e.printStackTrace();
//			size = -1;
//		}
//		List<RatingVector> vRatings = size <= 0 ? Util.newList() : Util.newList(size);
//		FetcherUtil.fillCollection(vRatings, vFetcher, false);
//		try {
//			vFetcher.reset();
//		}
//		catch (Exception e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
//		if (vRatings.size() < 2) return Util.newMap();
//		
//		int n = vRatings.size();
//		Map<Integer, Map<Integer, Double>> simMatrix = Util.newMap(n);
//		for (int i = 0; i < n; i++) {
//			RatingVector v1 = vRatings.get(i);
//			if (v1 == null || v1.size() == 0) continue;
//			
//			for (int j = i; j < n; j++) {
//				RatingVector v2 = vRatings.get(j);
//				if (v2 == null || v2.size() == 0) continue;
//				
//				double sim = cf.similar(v1, v2, null, null);
//				if (!Util.isUsed(sim)) continue;
//				
//				if (simMatrix.containsKey(v1.id()))
//					simMatrix.get(v1.id()).put(v2.id(), sim);
//				else {
//					Map<Integer, Double> map = Util.newMap(n);
//					simMatrix.put(v1.id(), map);
//					map.put(v2.id(), sim);
//				}
//				
//				if (i != j) {
//					if (simMatrix.containsKey(v2.id()))
//						simMatrix.get(v2.id()).put(v1.id(), sim);
//					else {
//						Map<Integer, Double> map = Util.newMap(n);
//						simMatrix.put(v2.id(), map);
//						map.put(v1.id(), sim);
//					}
//				}
//			}
//		}
//		
//		vRatings.clear();
//		return simMatrix;
//	}

	
	@Override
	public Inspector getInspector() {
		// TODO Auto-generated method stub
		return EvaluateGUI.createInspector(this);
	}

	
	@Override
	public DataConfig createDefaultConfig() {
		// TODO Auto-generated method stub
		DataConfig config = super.createDefaultConfig();
		config.put(VALUE_BINS_FIELD, VALUE_BINS_DEFAULT);
		config.put(BCF_MEDIAN_MODE_FIELD, BCF_MEDIAN_MODE_DEFAULT);
		config.put(MU_ALPHA_FIELD, MU_ALPHA_DEFAULT);
		config.put(SMTP_LAMBDA_FIELD, SMTP_LAMBDA_DEFAULT);
		config.put(SMTP_GENERAL_VAR_FIELD, SMTP_GENERAL_VAR_DEFAULT);
		config.put(TA_NORMALIZED_FIELD, TA_NORMALIZED_DEFAULT);
		
		return config;
	}


}
