#### Product Analyst - Technical Assessment

Thank you for taking time to complete this assessment. The assessment can be completed in either Python or SQL. It should not take more than 60 minutes to work on it (you can take an extra 5 minutes to read through the task description and instructions and whatever time is needed to prepare your working environment).

The main thing we are looking for is that you are careful and detail-oriented in your analysis.
Since on the job you will have the internet and additional resources to assist you in your work,
feel free to look up anything which allows you to better answer the questions. The exception,
of course, is that you cannot ask other people to help you directly and the work you submit
should be yours alone. Relatedly, do not share the task or the data afterwards in any way.
Please submit the completed assessment either as a Jupyter notebook with all calculations and
outputs clearly shown, or as an SQL script with an additional document that displays the
output of your script along with your insights and recommendations.

For this assessment you will use data from two .csv files. These files can be joined on the
common **user_id** column. The data in these files come from a sample of our English platform
student users who created an account in 2019. One file has profile information about the user
and the other has information about trial activity and conversion. The values in the column
class are calculated based on the graduation year that the users provide at account creation.

The values in the column **marketing_source** are user responses (chosen from a drop-down
menu) to how they heard about Amboss. The column **date_of_first_purchase** shows the date
the user converted and is missing if they have not converted yet. The column
**purchased_lifetime** is a boolean for whether their first purchase was the student lifetime
product. The columns ending with **_first_5_days** contain the number of chapters read, searched
made, and questions answered during the user’s trial period. The other columns should be
self-explanatory upon inspection.

Amboss has divided the English platform into two main markets: the United States (US) and
the rest of the world (RoW). A major part of your job here at Amboss would be to understand
the company’s performance in each market and how they compare with each other.
- We want to look at 30-day conversion rates (conversion within 30 days after
registration). Please provide an analysis and guide us through your insights about
30-day conversion rates for different markets (US, RoW). Please also provide possible
recommendations to stakeholders based on this information.