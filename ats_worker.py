import re

def ats_scorer(resume_text):
  """
  ATS scorer that rates a resume based on various points and gives a percentage as a score for : Sections, Brevity, Impact, and Styles.

  Args:
    resume_text (str): The text of the resume.

  Returns:
    dict: A dictionary containing the scores for each section.
  """

  # Define the sections that the ATS will look for.
  sections = ['Summary', 'experience', 'education', 'skills']

  # Define the scoring criteria for each section.
  section_scores = [
    {'name': 'summary', 'max_score': 10, 'criteria': ['Length (less than 100 words)', 'Use of keywords', 'Clear and concise language']},
    {'name': 'experience', 'max_score': 20, 'criteria': ['Relevant experience', 'Quantified accomplishments', 'Use of strong action verbs']},
    {'name': 'education', 'max_score': 10, 'criteria': ['Relevant degrees and certifications', 'GPA (if high)', 'Honors and awards']},
    {'name': 'skills', 'max_score': 10, 'criteria': ['In-demand skills', 'Variety of skills', 'Proficiency levels']},
    # {'name': 'Awards', 'max_score': 5, 'criteria': ['Relevant awards', 'Recent awards', 'Prestige of awards']}
  ]

  # Initialize the scores for each section.
  scores = {section: 0 for section in sections}

  # Check for each section in the resume.
  for section in sections:
    # Check if the section is present in the resume.
    if section in resume_text:
      # Get the text of the section.
      section_text = re.search(fr'^{section}(.*?)(?=\S+\n|\Z)', resume_text, flags=re.DOTALL).group(1)

      # Check if the section meets the scoring criteria.
      for criterion in section_scores[section]['criteria']:
        if criterion in section_text:
          # Increment the score for the section.
          scores[section] += section_scores[section]['max_score'] / len(section_scores[section]['criteria'])

  # Calculate the total score.
  total_score = sum(scores.values())

  # Calculate the percentage score.
  percentage_score = total_score / sum([section['max_score'] for section in section_scores]) * 100

  # Return the scores.
  return {
    'sections': scores,
    'total': total_score,
    'percentage': percentage_score
  }

# Example usage
resume_text = """
dev machine learning engineer faridabad haryana devparker gmail com linkedin github twitter kaggle core skills python javascript agile leadership scrum master computer vision generative ai git github mlops kubernetes mongodb firebase db pinecone mysql machine learning deep learning google cloud platform microsoft azure vector db docker flask fastapi professional experience dataknobs ml engineer june present automated key processes kreatewebsite major product enhance efficiency uplifted machine learning project predictive model precision made usable converting old written functions new one collaborate senior developers update website create new features open source contributor mlflow contributor august august contributed key functionality got merged administrator mlflow google cloud google cloud facilitator may july acquired proficiency docker mlops kubernetes kubernetes relevant projects sign language tutor march present used learning sign language fun interactive way chakla controller asphalt january january innovative racing game controlled unique physical interface round flat board blue square uses opencv computer vision techniques translate board movements game actions medsarthi january january helping seniors understand medications simple image upload voice enabled explanations education maharishi dayanand university rohtak bachelor computer science artificial intelligence rajokari institute technology dseu diploma information technology enabled service management
"""

scores = ats_scorer(resume_text)
print(scores)