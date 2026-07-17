# Review bộ probe hallucination/OOD

Bộ probe có 30 ảnh, được chọn trước khi retrieval bằng seed 42. Không bỏ ảnh chỉ vì caption hoặc retrieval không đẹp; nếu annotation gốc nhiễu, đánh dấu `revise` và ghi lý do trước khi chạy model.

Quyết định chung: `[ ] accept nguyên bộ` / `[ ] revise manifest và ghi lại quy tắc`.

| Category | Probe | Query image | Original caption | Review |
|---|---|---|---|---|
| dress | dress_q01 | <img src="../../data/fashioniq/images/B00ANK6ND0.jpg" width="110"> | is lighter and has longer sleeves [SEP]  off the shoulder and less fitted | `[ ] accept` / `[ ] revise` |
| dress | dress_q02 | <img src="../../data/fashioniq/images/B004O6LVY0.jpg" width="110"> | The dress in black in color. [SEP] is strapless and black | `[ ] accept` / `[ ] revise` |
| dress | dress_q03 | <img src="../../data/fashioniq/images/B004PKZWNG.jpg" width="110"> | is blue in color [SEP] Has long sleeves and a blue pattern | `[ ] accept` / `[ ] revise` |
| dress | dress_q04 | <img src="../../data/fashioniq/images/B00FBRGJKM.jpg" width="110"> | is tighter and fancier [SEP] tighter skirt and v neck | `[ ] accept` / `[ ] revise` |
| dress | dress_q05 | <img src="../../data/fashioniq/images/B0029YQQYE.jpg" width="110"> | more purple [SEP] is longer and more maroon | `[ ] accept` / `[ ] revise` |
| dress | dress_q06 | <img src="../../data/fashioniq/images/B00DFN4MSU.jpg" width="110"> | is shorter [SEP] is shorter and darker | `[ ] accept` / `[ ] revise` |
| dress | dress_q07 | <img src="../../data/fashioniq/images/B002WV0S6G.jpg" width="110"> | is shorter and more casual [SEP] is a lot shorter in tan coloring and the straps aren't similar. | `[ ] accept` / `[ ] revise` |
| dress | dress_q08 | <img src="../../data/fashioniq/images/B0059817JQ.jpg" width="110"> | has three colors [SEP] has more colors | `[ ] accept` / `[ ] revise` |
| dress | dress_q09 | <img src="../../data/fashioniq/images/B00B1YKB26.jpg" width="110"> |  has a  u shaped neck with red heels [SEP]  black and gray with no shoulder straps | `[ ] accept` / `[ ] revise` |
| dress | dress_q10 | <img src="../../data/fashioniq/images/B00BM82UCK.jpg" width="110"> | is thigh high and blue [SEP] Is blue and denim | `[ ] accept` / `[ ] revise` |
| shirt | shirt_q01 | <img src="../../data/fashioniq/images/B003JY6WY2.jpg" width="110"> | is gray colored with green and yellow design [SEP] Is grey in color | `[ ] accept` / `[ ] revise` |
| shirt | shirt_q02 | <img src="../../data/fashioniq/images/B0051D0X2Q.jpg" width="110"> | is blue plaid with white button shirt [SEP] is lighter | `[ ] accept` / `[ ] revise` |
| shirt | shirt_q03 | <img src="../../data/fashioniq/images/B00B31W2TC.jpg" width="110"> | is lighter with shorter sleeves [SEP] is less striped | `[ ] accept` / `[ ] revise` |
| shirt | shirt_q04 | <img src="../../data/fashioniq/images/B002UQJENQ.jpg" width="110"> | is a black t shirt [SEP] has text in red | `[ ] accept` / `[ ] revise` |
| shirt | shirt_q05 | <img src="../../data/fashioniq/images/B005CQAIYK.jpg" width="110"> | is a lighter color and has shorter sleeves [SEP] Is white with shorter sleeves. | `[ ] accept` / `[ ] revise` |
| shirt | shirt_q06 | <img src="../../data/fashioniq/images/B00B1FCGFU.jpg" width="110"> | is blue with image of dolphins [SEP] is closer to blue | `[ ] accept` / `[ ] revise` |
| shirt | shirt_q07 | <img src="../../data/fashioniq/images/B0095VZ8JI.jpg" width="110"> | Is less animal-like and more wordy [SEP] has a white chest logo | `[ ] accept` / `[ ] revise` |
| shirt | shirt_q08 | <img src="../../data/fashioniq/images/B00AOJHO56.jpg" width="110"> | Has a Hawaii summery feel with more colors [SEP] has more colors and more graphics | `[ ] accept` / `[ ] revise` |
| shirt | shirt_q09 | <img src="../../data/fashioniq/images/B001CWN1A6.jpg" width="110"> | is gray with long sleeves [SEP] is darker | `[ ] accept` / `[ ] revise` |
| shirt | shirt_q10 | <img src="../../data/fashioniq/images/B00AZIAZK2.jpg" width="110"> | Is more whimsical and black [SEP] is a dark t-shirt with a graphic | `[ ] accept` / `[ ] revise` |
| toptee | toptee_q01 | <img src="../../data/fashioniq/images/B008BT599E.jpg" width="110"> | The shirt is black with the word DADD. [SEP] Is a regular t-shirt with different graphic. | `[ ] accept` / `[ ] revise` |
| toptee | toptee_q02 | <img src="../../data/fashioniq/images/B00BB9UBNU.jpg" width="110"> | is black and has longer sleeves [SEP] Has longer sleeves and a tighter neckline | `[ ] accept` / `[ ] revise` |
| toptee | toptee_q03 | <img src="../../data/fashioniq/images/B00BF6VD2M.jpg" width="110"> | is white with longer sleeves [SEP]  with v neckline | `[ ] accept` / `[ ] revise` |
| toptee | toptee_q04 | <img src="../../data/fashioniq/images/B00A3QE0QQ.jpg" width="110"> | is lighter [SEP] has shorter sleeves and more blue | `[ ] accept` / `[ ] revise` |
| toptee | toptee_q05 | <img src="../../data/fashioniq/images/B007TLO9D2.jpg" width="110"> | is more witty and less inspirational [SEP] is green and with new print. | `[ ] accept` / `[ ] revise` |
| toptee | toptee_q06 | <img src="../../data/fashioniq/images/B004U8FZE4.jpg" width="110"> | is dark brown and has sleeves [SEP] Has longer sleeves and a darker pattern | `[ ] accept` / `[ ] revise` |
| toptee | toptee_q07 | <img src="../../data/fashioniq/images/B00C85FTH4.jpg" width="110"> | a button up and much longer [SEP] is buttoned and has collar | `[ ] accept` / `[ ] revise` |
| toptee | toptee_q08 | <img src="../../data/fashioniq/images/B003EACZ9M.jpg" width="110"> | is yellow colored [SEP] is yellow | `[ ] accept` / `[ ] revise` |
| toptee | toptee_q09 | <img src="../../data/fashioniq/images/B003IKOOTW.jpg" width="110"> | is black colored with sleeves [SEP] is black with white print and full sleeves on the crew neck. | `[ ] accept` / `[ ] revise` |
| toptee | toptee_q10 | <img src="../../data/fashioniq/images/B0094KPUK2.jpg" width="110"> | is longer and has longer sleeves [SEP] longer | `[ ] accept` / `[ ] revise` |

## Ghi chú review

- `<PLACEHOLDER_REVIEWER>`:
- `<PLACEHOLDER_DATE>`:
- `<PLACEHOLDER_AMBIGUOUS_PROMPTS_AND_DECISION>`:
