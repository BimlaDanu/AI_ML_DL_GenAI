Prognostica Data Science Challenge — English Translation

Tasks

Please prepare two presentations for the following tasks:


First presentation (~10 minutes, in German): Present your solution as you would like to introduce it to us.
Second presentation (~5 minutes, in English, covering Tasks 2–3 only): Present your solution to the management of a fictitious prospective client.


All required files are provided as attachments. A Windows PC with USB port and installed Microsoft Office and Adobe Reader will be available for the presentation.


Task 1 — Data Preparation

In the file daten_kunden.csv you will find a table with revenue figures for several customers of the fictitious company BFCC AG. The data must first be prepared as follows:


All customers whose ID ends with "DEK" must be removed.
The remaining customer values must be aggregated on a monthly basis, so that only one total value per month exists — the sum of all remaining customer values.


Please find a reproducible solution that could be applied to a new dataset of the same structure without any additional effort.


Task 2 — "Past Values for the Future"

The prepared monthly figures from Task 1 now contain the total revenue figures of interest.


Visualise the data.
Forecast the next 6 months.
Present your results graphically.



Task 3 — "Changes in the Environment"

In the file afo.csv you will find monthly values of the fictitious AFO Business Climate Index. The client considers this index important for their business sector.

Would you revise your forecast from Task 2 based on knowledge of this index?


Data Files

|File|Contents|
|---|---|
|daten_kunden.csv|Monthly revenue rows per customer (date, customer_id, revenue)|
|afo.csv|Monthly AFO Business Climate Index values (date, afo_index)|

Deliverables

- Reproducible code (Python or R) with clear documentation
- Two presentations as described above
- Visualisations for all three tasks
