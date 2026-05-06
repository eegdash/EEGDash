# Evidence-based architecture for a Python library PyData tutorial

## Executive Summary

This report starts from a practical premise: a good tutorial for a Python library in the PyData context needs to function simultaneously as**teaching material**, **reproducible software artifact**and**onboarding experience**for the community. The most robust literature to support this design does not come from a single field, but from the convergence of cognitive psychology, STEM education, educational computer science, and, in more specific points, memory neuroscience. For a generic audience of adults or university students in scientific computing, the strongest evidence favors**recovery practice**, **spacing**, **solved examples**, **rapid formative feedback**, **active engagement**and**complementary multimodality**; **interleaving**and**self-explanation**They are also promising, but depend more on the type of content and the task.[\[1\]](https://www.whz.de/fileadmin/lehre/hochschuldidaktik/docs/dunloskiimprovingstudentlearning.pdf)

The central recommendation, therefore, is an architecture.**notebook-first, but not notebook-only**The goal is to write the tutorial in notebooks to preserve narrative, visualization, and interactive execution; pair them with text files using Jupytext for versioning and review; publish the navigable version in Jupyter Book; and ensure clean and parameterizable execution with nbclient and papermill. This arrangement leverages the pedagogical value of notebooks—combining executable code, narrative text, visualizations, and exploration—without passively accepting their typical problems of hidden state, order-of-execution, and low reproducibility.[\[2\]](https://users.ugent.be/~gvhaelew/doc/jupyter-edu-book.pdf)

For a PyData-style tutorial, the most defensible format is a**hands-on autocontido**With materials provided before the session, a strong emphasis on short exercises, and a reproducible environment via repository, Binder, and optionally Colab. A recent PyData conference CFP describes tutorials as self-contained, 90-minute hands-on sessions with pre-distribution of materials and explicit requirements; this aligns well with the PyData program's identity as an educational forum for the data analytics community. In parallel, the community requires adherence to the NumFOCUS code of conduct.[\[3\]](https://onderwijs.felienne.nl/vakdidactiek/materiaal/sweller_worked_examples.pdf)which should influence the language, examples, accessibility, and moderation of the session.[\[4\]](https://numfocus.org/programs/pydata)

In terms of instructional design, the tutorial should alternate:**short solved example**, **microchecking comprehension**, **active coding segment**, **immediate feedback**and**spaced revisit**Key concepts are introduced throughout the session. Instead of a long expository sequence, the literature recommends formative checks every 10–15 minutes, predictions before execution, and tasks that require the learner to recall the API and flow logic rather than simply recognizing the solution on the screen. This is particularly important for Python libraries, because the most common mistake in “pretty” tutorials is confusing visual fluency with real learning.[\[5\]](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006915)

Validation also needs to move beyond the subjective realm. An agent can measure tutorial quality with a relatively objective battery of tests:**structural rubric**, **clean execution**, **reproducibility in clean kernel**, **unit and visual tests**, **rolling the notebook**, **time and memory budgets**, **accessibility checks**and**engagement proxies**Based on event logs and small pre/post assessments. The important point is not to treat "engagement" as synonymous with learning: the literature in learning analytics recommends combining behavioral traits with performance measures and exercising special caution with time-on-task metrics, because the estimation method changes the results.[\[6\]](https://u.osu.edu/yuan.818/files/2020/09/Measuring-student-engagement-in-technology-mediated-learning-A-review-w-notes1.pdf)

The operational conclusion is straightforward: if you're delegating tutorial creation to an agent, ask for a three-layer deliverable. The first layer is pedagogical: objectives, solved examples, exercises, recovery checkpoints, spaced repetition, and feedback. The second is technical: paired notebooks, a reproducible environment, tests, CI, and publication. The third is evidence-based: an automated tutorial quality score, execution logs, and a final report that clearly states the objectives.**what happened**, **what went wrong**and**why**. [\[7\]](https://docs.github.com/actions/guides/building-and-testing-python)

## Scientific basis

The safest interpretation from the literature is that the causal basis for effective tutorials comes first from**cognitive psychology and education**...and only after neuroscience. In other words: neuroscience helps explain mechanistic plausibility—for example, working memory limits, consolidation, and hippocampal strengthening—but the instructional recommendations most directly applicable to tutorials have been tested primarily in behavioral studies and educational meta-analyses. This is a strength, not a weakness: for tutorial design, what matters most is what changes performance, retention, and transfer in real-world tasks.[\[8\]](https://www.researchgate.net/publication/291826702_The_right_time_to_learn_Mechanisms_and_optimization_of_spaced_learning)

### Solved examples, sub-objectives, and fading

Solved examples are one of the most stable pieces of literature. In the classic 1985 algebra study, students who studied solved examples processed problems in less time and made fewer errors than groups focused on conventional problem-solving; the interpretation was that examples reduce unproductive searching and help form schemas. Later revisions transformed this into a principle of instructional design: examples are particularly useful early in skill acquisition, when the cognitive load of “figuring out the next step” competes with learning the problem structure itself. A recent meta-analysis in mathematics concludes that solved examples, alone or paired with similar practice, are**moderately effective**to improve performance throughout the school years.[\[9\]](https://onderwijs.felienne.nl/vakdidactiek/materiaal/sweller_worked_examples.pdf)

The translation for a Python library tutorial is very concrete. Instead of presenting an entire API at once, the tutorial should break down the flow into named and reusable sub-goals — for example:**load data**, **inspect structure**, **apply transformation**, **execute estimator**, **evaluate result**, **view output**The literature on teaching programming explicitly recommends the use of solved examples with...**subgoals labels**This is precisely because beginners do not spontaneously perceive the common structure between isomorphic problems. In a scientific library, this reduces the chance of the learner memorizing a sequence of calls without understanding the conceptual pattern that they can reuse later.[\[10\]](https://journals.plos.org/ploscompbiol/article%3Fid%3D10.1371/journal.pcbi.1006023)

The corollary is the**support fading**The first notebook can show the entire flow; the second can hide small sections; the third should require the learner to recognize the correct sub-goal and write part of the solution. The review of worked examples and the theoretical framework for skill acquisition itself support the idea that examples are better in the initial phases, but that, later on, practice with increasing autonomy becomes superior. In tutorials, this means: not leaving the student forever in "Shift+Enter" mode.[\[11\]](https://mrbartonmaths.com/resourcesnew/8.%20Research/Explicit%20Instruction/Structuring%20the%20Transition%20From%20Example%20Study%20to%20Problem%20Solving.pdf)

### Recovery practices, immediate feedback, and predictions.

Among the best-supported techniques, recovery practice occupies a central place. Dunlosky's review classifies**practice testing**as a highly useful technique; Rowland's meta-analysis shows that testing, rather than just rereading, produces a retention advantage; and Roediger and Butler's review summarizes decades of research showing that retrieving information from memory often produces more long-term learning than repeated study. In a landmark experiment in science education, retrieval outperformed elaborative study by concept mapping in measures of meaningful learning and transfer, even when study time was equalized.[\[12\]](https://www.whz.de/fileadmin/lehre/hochschuldidaktik/docs/dunloskiimprovingstudentlearning.pdf)

For library tutorials, "retrieval practice" should not be interpreted as "giving a test." It can appear in small, frequent forms: asking the learner to...**predict the departure**before running the cell; ask them to write a memory method call before showing the template; ask them to rebuild a short pipeline without consulting the previous cell; or use a 1–2 minute quiz at the end of each block. This pattern aligns with programming teaching recommendations that advocate frequent formative checks, prediction questions, and short coding activities every 10–15 minutes.[\[13\]](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1007433)

The critical point is the**feedback**Hattie and Timperley's review shows that feedback is a powerful influence, but not uniformly beneficial; it needs to be well-configured. A 2020 meta-analysis revisiting the topic, with 435 studies and 994 effects, reinforces that feedback improves learning, but with strong heterogeneity. Dunlosky's own review is specific: retrieval practice.**com feedback**It is more robust than recovery practice alone. In a Python tutorial, this favors visible testing (assert), interpretable error messages, localized hints, and automatic checks that indicate**What to correct**in the task, not just "right/wrong".[\[14\]](https://conselhopedagogico.tecnico.ulisboa.pt/files/sites/32/hattie-and-timperley-2007.pdf)

Neuroscience helps explain why this combination works. A review on spacing shows plausible cellular and molecular mechanisms for spaced training; a neuroimaging study on retrieval practice found enhanced processing in the anterior and posterior hippocampus; and reviews in cognitive neuroscience of motivation highlight the role of dopaminergic reward-predicting signals in learning. The prudent inference for tutorials is:**recover, make small-scale errors, and receive quick correction.**This likely creates a more favorable dynamic for consolidation than passively watching a flawless demonstration.[\[15\]](https://www.researchgate.net/publication/291826702_The_right_time_to_learn_Mechanisms_and_optimization_of_spaced_learning)

### Spacing and interleaving

The literature on spacing is particularly robust. Cepeda's classic meta-analysis compiled 839 reviews and found an advantage to distributed practice; Dunlosky classified**distributed practice**as a highly useful technique; and Smolen's mechanistic review summarizes why intervals between repetitions favor more stable memory than massive training. For tutorials, this means that essential library concepts should not appear just once. The same core API—for example, an objectDataset, a function offit, or a method ofplot— It should be revisited after a break filled with another activity.[\[16\]](https://augmentingcognition.com/assets/Cepeda2006.pdf)

Interleaving has more nuanced support. Dunlosky classified the technique as having moderate utility, and Brunmair and Richter's meta-analysis found a moderate positive effect of interleaving, but also showed important material and context dependence; the article itself warns of careful use with expository texts and certain word categories. In terms of library tutorials, the most promising application is interleaving.**similar tasks that require strategic discrimination**For example, switching between two types of graphs that look similar but require different APIs, or between two estimators with similar signatures but different assumptions. The worst use would be to interleave content without a common structure, producing only confusion.[\[17\]](https://www.whz.de/fileadmin/lehre/hochschuldidaktik/docs/dunloskiimprovingstudentlearning.pdf)

The strongest case is**spaced retrieval**Distributed retrieval over time combines two desirable difficulties. The meta-analysis by Latimier and colleagues, based on 29 studies, precisely frames this arrangement as a combination of retrieval and spacing. A well-designed tutorial, therefore, not only revisits concepts; it compels the learner to...**I rebuilt them**at later points in the session or series of notebooks.[\[18\]](https://www.lscp.net/persons/ramus/docs/EPR20.pdf)

### Active coding, live coding, and constructive engagement

The difference between “seeing” code and “doing” code is supported by two converging lines of thought. The first is general: the ICAP framework proposes a hierarchy of engagement in which passive activities tend to produce less learning than active, constructive, and interactive activities. The second is specific to STEM: the meta-analysis by Freeman et al. in 225 studies found an average increase of 0.47 standard deviations in performance and a relevant reduction in failure rates under active learning compared to traditional lectures.[\[19\]](https://education.asu.edu/sites/g/files/litvpz656/files/lcl/chiwylie2014icap_2.pdf)

In programming education, this translates well to**live coding**, **code-along**, **peer instruction**and**pair programming**A study on live coding with beginners found three benefits perceived by students: making the programming process more understandable, supporting debugging learning, and exposing good practices; the same study observed that students preferred to follow along by coding, rather than just observing. In parallel, programming teaching guides advocate for predicting code behavior, peer instruction to detect and correct misconceptions, and pair programming to reduce skill level asymmetry and make mutual support scalable.[\[20\]](https://pages.cs.wisc.edu/~gerald/papers/LiveCoding.pdf)

For a scientific library, the implication is simple: the tutorial should not resemble documentation read aloud. The person who creates the material should include moments where the learner...**editor**, **execute**, **failure**, **corrected**and**discusses**A good test for this is to ask: "If I remove all the exercise cells, does the tutorial still 'work'?" If the answer is yes, it's probably too expository.[\[21\]](https://education.asu.edu/sites/g/files/litvpz656/files/lcl/chiwylie2014icap_2.pdf)

### Multimodal content and cognitive load

Multimodality helps, but under one crucial condition: the channels need to be**complementary**...non-redundant. Mayer defines multimedia learning as learning from words and images, and his line of research documents various design effects; Wilson summarizes the principle operationally by recommending the integration of visual and linguistic information; and a 2015 commentary concludes that conditions for effectiveness are already relatively clear for basic multimedia materials, although more complex environments—games, rich simulations, intelligent tutors—still have more open moderators. A 2025 meta-analysis reinforces this point by showing that the strongest effects came less from cosmetic design adjustments and more from...**active interventions**in multimedia materials.[\[22\]](https://www.jsu.edu/online/faculty/MULTIMEDIA%20LEARNING%20by%20Richard%20E.%20Mayer.pdf)

In PyData tutorials, well-used multimodality means: a graph accompanied by a short interpretation; a small table to anchor values; a simple diagram of the object flow; and text that explains it.**because**That output matters. What should be avoided is excessive duplication, such as narrated videos with identical subtitles and blocks of text reproducing what is already visible in the graph. The rule of thumb is: each format should carry information that aids inference or reduces search, not compete for the same attention.[\[23\]](https://www.jsu.edu/online/faculty/MULTIMEDIA%20LEARNING%20by%20Richard%20E.%20Mayer.pdf)

### Note on literature in Portuguese

There is useful literature in Portuguese to contextualize and translate these recommendations for Portuguese-speaking teachers, but it is less abundant and, in general, less experimental than the English-language corpus. For remedial practice, there is a Brazilian systematic review of the testing effect and articles in Portuguese discussing factors relevant to educators and test formats in the school environment. For the neuroeducation axis, there are reviews in Portuguese on neuroscience and learning, neuroplasticity, memory, and the contributions of neuroeducation. These sources are valuable for teacher training and communication with Brazilian teams, but, for effect estimates and fine-tuned tutorial design, the quantitative core continues to come mainly from the English-language literature.[\[24\]](https://www.researchgate.net/publication/274359975_A_Systematic_Review_of_the_Testing_Effect_in_Learning)

## Translation for drawing a PyData tutorial

The PyData context matters because it sets expectations for format, tone, and reuse. The PyData program is presented as an international community educational forum for sharing ideas and learning around data analytics tools; a recent CFP for PyData London 2026 describes tutorials as self-contained, 90-minute hands-on sessions with pre-distributed materials and clearly specified requirements. Furthermore, events require adherence to the NumFOCUS code of conduct. This creates four design requirements:**self-restraint**, **prior clarity of the environment**, **reusable material after the event**and**inclusive community language**. [\[4\]](https://numfocus.org/programs/pydata)

At the same time, notebooks remain the best medium for “telling a computational story” with code, prose, and visualization. The handbook*Teaching and Learning with Jupyter*This describes exactly that: notebooks combine executable code, equations, visualizations, and narrative text; they can be used for demonstrations, interactive labs, supplementary materials, assignments, and entire courses; and they allow for incremental adoption. The problem is that research on notebooks itself shows real risks to reproducibility and maintainability, including hidden state, out-of-order cells, hard-coded paths, and poorly specified dependencies. The best compromise for a library tutorial is, therefore, to use notebooks for teaching and...**pairing textual**for quality engineering.[\[25\]](https://users.ugent.be/~gvhaelew/doc/jupyter-edu-book.pdf)

The comparison below summarizes this trade-off.

| Format | When to use | Main advantages | Main risks | Trial for PyData tutorial |
| :---- | :---- | :---- | :---- | :---- |
| Single notebook (.ipynb) | Quick prototype, small workshop | Excellent for storytelling, visualization, and interactive performance. | Difficult to revise in diff, more prone to hidden state and style drift. | **Acceptable for draft; insufficient as a final artifact.** |
| Notebook paired with.pyor.mdvia Jupytext | Main development | It maintains interactivity and improves versioning, review, and refactoring. | It requires discipline in synchronization and metadata conventions. | **Best balance point** |
| Collection of notebooks published in Jupyter Book | Public distribution, durable documentation/tutorial | Navigation, TOC, references, continuous publishing, integration with online execution. | Extra layer of build and maintenance | **Ideal publication format** |
| Script \+ static documentation | API reference or internal automation | Clean differences, good modularity, simple IC | It loses the immediacy and pedagogical affordances of notebooks. | **Good as supplementary material; weak as a main tutorial.** |
| Colab-first | Public with high installation friction | Zero setup, easy access via browser. | Reproducibility and resources not guaranteed; less control over the environment. | **Good as an entry point, not as a canonical source.** |

This synthesis is derived from the literature on notebooks in education, the study on quality and reproducibility in notebooks, the comparison between notebooks and scripts, the Jupytext documentation, and the Jupyter Book publishing resources.[\[26\]](https://users.ugent.be/~gvhaelew/doc/jupyter-edu-book.pdf)

For a Python library, the recommended architecture is as follows. The master tutorial should be a**short series of notebooks**The repository should contain: an onboarding and minimal setup notebook; a main flow notebook with a solved example and exercise; a transfer notebook with variations, common errors, and a mini-project; and a version published in Jupyter Book.environment.ymlor equivalent, a small, versioned dataset, fixed seeds, and secondary links to Binder and Colab. Binder favors reproducibility if the link points to a commit hash; Colab reduces input friction, but should not be the normative reference for the environment.[\[27\]](https://mybinder.readthedocs.io/en/latest/tutorials/reproducibility.html)

From a pedagogical point of view, the most robust template for 90 minutes is this: a short opening with objectives and prerequisites; a solved example with sub-objectives; a micro-check for remediation; a guided exercise with feedback; a second block interspersing a similar, but not identical, case; a short mini-project; and a closing with a remediation checklist and next steps. The important part is not the exact duration of each block, but the alternation between them.**demonstration**, **recovery**, **production**and**correction**. [\[28\]](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006915)

## Operational checklist

The checklist below is the "serious minimum viable product" for a PyData-style Python library tutorial. It summarizes recommendations for pedagogical research, notebook best practices, web accessibility, community standards, and publication/reproducibility guidelines.[\[29\]](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006915)

| Dimension | Minimum criteria | A sign of excellence | Risk if missing |
| :---- | :---- | :---- | :---- |
| Audience | Explicit persona, prerequisites, and stated exit objective. | Measurable objectives aligned with the final evaluation. | Material that is "neither for beginners nor advanced players" |
| Structure | H1 initial, objectives, setup, dataset, exercises, recap | Navigable TOC in Jupyter Book, estimated time per block. | Loss of orientation and poor pacing. |
| Examples | At least one example solved by a key concept. | Labeled sub-objectives and fading support throughout the series. | Superficial imitation without transfer. |
| Recovery | Short check every 10–15 min | Predictions before execution and final memory recap | The illusion of fluency |
| Spacing | Central concepts reappear later | They reappear in a new context and with active production. | Fragile and localized learning |
| Interleaving | Similar variations appear side by side. | Contrasting cases to discriminate the correct strategy. | Unproductive confusion or overly repetitive blocks. |
| Feedback | asserts or visible criteria in the exercises | Specific feedback, gradual tips, and task-oriented messages. | Silent error or opaque correction |
| Data | Small, licensed, versioned, and stable dataset. | Local cache, checksum, and "tiny" sample for CI. | Download failure, slow speed, non-deterministic results. |
| Reproducibility | Specified environment, clean kernel, top-down execution. | Binder by commit, paired notebooks, automatic publishing. | "It only works on the author's machine." |
| Accessibility | Correct headings, alt text in markdown images, textual summaries of graphs. | Alternative tables/summaries, clear language, and careful contrast. | Users with screen readers or low vision are left behind. |
| Community | Inclusive language and adherence to the code of conduct. | Culturally neutral examples and supporting documentation for the session. | Social noise, barriers to belonging |
| Post-event reuse | Public repository and execution instructions | Versioned artifacts, releases, changelog, and bibliography. | Tutorial “dies” after conference. |

## Automated validation in Python

Validating a tutorial shouldn't rely solely on informal human review. The ecosystem's official documentation already covers almost all the necessary pieces: nbclient for programmatic execution, papermill for parameterization, nbval for validating saved outputs, nbQA for lintling notebooks, nbgrader for test cells, and GitHub Actions for running everything in a version matrix. What's usually missing isn't the tool itself, but...**the design of an automated rubric**that combines technique, pedagogy, and UX.[\[30\]](https://nbclient.readthedocs.io/en/latest/client.html)

The following table summarizes a practical validation test.

| Method | What does it measure? | Tools | Good for | Main limitation | Suggested starting gate |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Structural heading | Presence of sections, pacing, exercises, objectives | nbformat, regex, tags | Explicit pedagogy | It does not measure actual learning. | 100% of the required sections |
| Notebook lint | Style, imports, code quality | nbqa, ruff, black | Editorial/technical consistency | False positives in exploratory cells | zero critical errors |
| Clean execution | Runtime errors in new kernel | nbclient | Detect hidden state | It does not measure educational usefulness. | zero errors |
| Parameterized execution | Robustness across different options/datasets | papermill | Reusable tutorial and demos | Slower | The "tiny" scenario should pass. |
| Exit validation | Discrepancy between expected and actual output. | nbval | Regression of stabilized notebooks | Non-deterministic outputs require sanitization. | Canonical notebooks pass |
| Correctness | API, shapes, values, contracts | pytest | Scientific/technical guarantee | Requires well-defined fixtures. | all essential tests passed |
| Visual regression | If plots have been changed improperly | matplotlib.testing | Libraries with strong visual output. | Sensitive to backend/sources | difference ≤ tolerance |
| Performance | Time and memory | subprocess, resource, psutil | Practical experience from the tutorial | Limits depend on the runner. | explicit budget |
| Accessibility | Headings, alt text, text summaries | parsing markdown/HTML | Inclusion and navigation | Not everything can be inferred automatically. | 100% markdown images with alt |
| Engagement/proxies | Time spent on task, retries, hints, pre/post earnings | pandas \+ logs | Piloting and continuous improvement | Proxy, not ground truth | use in conjunction with performance |

These recommendations are consistent with studies on notebook reproducibility, documentation of the Jupyter ecosystem, and learning analytics literature that recommends multiple measures and special caution regarding time-on-task and behavioral traits.[\[31\]](https://leomurta.github.io/papers/pimentel2019a.pdf)

### Structural and pedagogical rubric

The first layer of validation should be cheap and brutally simple: Does the notebook have the sections it should have? Are there enough exercises? Does the first markdown include an H1? Are there objectives, prerequisites, setup, and recap? Are there short checkpoints throughout the flow? This operationalizes the recommendations for reverse design, pacing through formative assessment, and accessible heading structure.[\[32\]](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006915)

from \_\_future\_\_ import annotations

import json  
import re  
from pathlib import Path

import nbformat

REQUIRED\_HEADINGS \= \[  
"Learning objectives"  
"Prerequisites"  
"Installation"  
    "Exercise",  
    "Recap",  
\]

def load\_notebook(path: str | Path):  
    with open(path, "r", encoding="utf-8") as f:  
        return nbformat.read(f, as\_version=4)

def text\_of(cell) \-\> str:  
    return cell.get("source", "").strip()

def validate\_structure(path: str | Path) \-\> dict:  
    nb \= load\_notebook(path)  
    cells \= nb.cells

    first\_cell \= cells\[0\] if cells else {}  
    first\_ok \= (  
        first\_cell.get("cell\_type") \== "markdown"  
        and bool(re.search(r"^\#\\s+\\S+", text\_of(first\_cell), flags=re.M))  
    )

    full\_text \= "\\n\\n".join(text\_of(c) for c in cells if c.get("cell\_type") \== "markdown")  
    missing \= \[h for h in REQUIRED\_HEADINGS if h.lower() not in full\_text.lower()\]

    exercise\_cells \= sum(  
        1 for c in cells  
        if c.get("cell\_type") \== "markdown"  
and re.search(r"\\b(exerc\[ií\]cio|fa\[aç\]a voc\[eê\]|tente agora|desafio)\\b", text\_of(c), re.I)  
    )

    code\_cells \= sum(1 for c in cells if c.get("cell\_type") \== "code")  
    markdown\_cells \= sum(1 for c in cells if c.get("cell\_type") \== "markdown")

    exercise\_density \= exercise\_cells / max(1, markdown\_cells \+ code\_cells)

    return {  
        "file": str(path),  
        "first\_h1\_ok": first\_ok,  
        "missing\_headings": missing,  
        "exercise\_cells": exercise\_cells,  
        "code\_cells": code\_cells,  
        "markdown\_cells": markdown\_cells,  
        "exercise\_density": round(exercise\_density, 3),  
        "pass": first\_ok and not missing and exercise\_cells \>= 3,  
    }

report \= validate\_structure("tutorials/intro.ipynb")  
print(json.dumps(report, ensure\_ascii=False, indent=2))

Expected output:

{  
  "file": "tutorials/intro.ipynb",  
  "first\_h1\_ok": true,  
  "missing\_headings": \[\],  
  "exercise\_cells": 5,  
  "code\_cells": 18,  
  "markdown\_cells": 14,  
  "exercise\_density": 0.156,  
  "pass": true  
}

### Clean execution, parameterization, and reproducibility

The second layer is running the notebook in**clean kernel**This is essential because notebooks have a well-documented history of failing when run from top to bottom in a new environment. The safest approach is to run them with nbclient and, when notebooks have alternative cases, parameterize them with papermill. For executable publication, Binder also recommends explicit reproducibility practices, such as freezing the environment and using commit hash in the link.[\[33\]](https://nbclient.readthedocs.io/en/latest/client.html)

from \_\_future\_\_ import annotations

import hashlib  
import json  
from pathlib import Path

import nbformat  
from nbclient import NotebookClient

def execute\_notebook(src: str, dst: str, timeout: int \= 180\) \-\> None:  
    with open(src, "r", encoding="utf-8") as f:  
        nb \= nbformat.read(f, as\_version=4)

    client \= NotebookClient(nb, timeout=timeout, kernel\_name="python3")  
    executed \= client.execute()

    with open(dst, "w", encoding="utf-8") as f:  
        nbformat.write(executed, f)

def stable\_output\_hash(path: str) \-\> str:  
    with open(path, "r", encoding="utf-8") as f:  
        nb \= nbformat.read(f, as\_version=4)

    payload \= \[\]  
    for cell in nb.cells:  
        if cell.cell\_type \!= "code":  
            continue  
        outputs \= \[\]  
        for out in cell.get("outputs", \[\]):  
            o \= dict(out)  
\# remove non-deterministic fields  
            o.pop("execution\_count", None)  
            if "metadata" in o:  
o\["metadata"\] \= {}  
            outputs.append(o)  
        payload.append(outputs)

    blob \= json.dumps(payload, sort\_keys=True, ensure\_ascii=False).encode("utf-8")  
    return hashlib.sha256(blob).hexdigest()

execute\_notebook("tutorials/intro.ipynb", "artifacts/run1.ipynb")  
execute\_notebook("tutorials/intro.ipynb", "artifacts/run2.ipynb")

h1 \= stable\_output\_hash("artifacts/run1.ipynb")  
h2 \= stable\_output\_hash("artifacts/run2.ipynb")

print({"hash\_run1": h1, "hash\_run2": h2, "reproducible": h1 \== h2})

Expected output:

{  
  'hash\_run1': 'a0c7...9f2',  
  'hash\_run2': 'a0c7...9f2',  
  'reproducible': True  
}

If the notebook uses parameters, the complementary test is to run it with a minimal dataset:

import papermill as pm

pm.execute\_notebook(  
"tutorials/modelagem.ipynb",  
"artifacts/modelagem\_tiny.ipynb",  
    parameters={  
        "sample\_frac": 0.05,  
        "random\_seed": 42  
    }  
)  
print("OK: Parameterized notebook executed.")

Expected output:

OK: Parameterized notebook executed.

### Correction tests, notebooks, and visual outputs.

Practical exercises should have automatic verification whenever possible. The nbgrader documentation recommends test cells withassertNBVAL runs notebooks and compares outputs; and, in programming contexts, automated evaluation platforms store multiple attempts, errors, and scores per test case, which is useful for both feedback and analytics. For libraries with a strong visual component, it's worth adding image regression.[\[34\]](https://nbgrader.readthedocs.io/en/latest/user_guide/creating_and_grading_assignments.html)

\# tests/test\_api\_example.py  
from \_\_future\_\_ import annotations

import numpy as np  
import pandas as pd

from my\_library import PipelineExample

def test\_fluxo\_basico():  
    df \= pd.DataFrame({  
        "x": \[1, 2, 3, 4\],  
"y": \[2, 4, 6, 8\]  
    })  
    pipe \= PipelineExemplo().fit(df\[\["x"\]\], df\["y"\])  
    pred \= pipe.predict(pd.DataFrame({"x": \[5, 6\]}))

    assert pred.shape \== (2,)  
assert np.allclose(pred, \[10, 12\], atol=1e-6)

Expected output:

1 passed in 0.42s

For visual regression:

from \_\_future\_\_ import annotations

from pathlib import Path

import matplotlib  
matplotlib.use("Agg")  
from matplotlib.testing.compare import compare\_images

def test\_plot\_regression():  
    baseline \= Path("tests/baseline/roc\_curve.png")  
    candidate \= Path("artifacts/roc\_curve.png")

    diff \= compare\_images(str(baseline), str(candidate), tol=2.0)  
    assert diff is None, diff

Expected output:

1 passed in 0.31s

### Performance budget

Educational quality drops rapidly when the notebook takes too long to run or consumes too much memory. In Computer Science, this needs to become an explicit budget requirement. The exact thresholds are heuristic and vary by domain, but the principle is objective: the tutorial should run on common infrastructure, with a small dataset, in a time compatible with a workshop, and without requiring a high-end machine.[\[35\]](https://leomurta.github.io/papers/pimentel2019a.pdf)

from \_\_future\_\_ import annotations

import json  
import subprocess  
import time  
from pathlib import Path

def run\_with\_budget(notebook: str, max\_seconds: int \= 180\) \-\> dict:  
    start \= time.perf\_counter()  
    proc \= subprocess.run(  
        \[  
            "python", "-m", "jupyter", "nbconvert",  
            "--to", "notebook",  
            "--execute", notebook,  
            "--output", "executed.ipynb",  
            "--ExecutePreprocessor.timeout=180",  
        \],  
        capture\_output=True,  
        text=True,  
    )  
    elapsed \= time.perf\_counter() \- start

    return {  
        "returncode": proc.returncode,  
        "elapsed\_seconds": round(elapsed, 2),  
        "pass": proc.returncode \== 0 and elapsed \<= max\_seconds,  
"stderr\_tail": proc.stderr.splitlines()\[-10:\],  
    }

report \= run\_with\_budget("tutorials/intro.ipynb", max\_seconds=120)  
print(json.dumps(report, indent=2, ensure\_ascii=False))

Expected output:

{  
  "returncode": 0,  
  "elapsed\_seconds": 38.47,  
  "pass": true,  
  "stderr\_tail": \[\]  
}

### Engagement metrics and learning proxies

Engagement is not observable by a single metric. Henrie's review shows that different approaches capture different dimensions of the construct; Baker and Siemens remind us that behavioral detectors can be useful, but need to be interpreted carefully; and studies on time-on-task show that the very way of estimating the variable alters conclusions. In programming, there is evidence that indicators such as compilation error repetitions, retries, keystrokes, and finer versions of time-on-task can better predict difficulties and performance than coarse measures. Therefore, the ideal is to use a**proxy panel**, never a single number.[\[36\]](https://u.osu.edu/yuan.818/files/2020/09/Measuring-student-engagement-in-technology-mediated-learning-A-review-w-notes1.pdf)

from \_\_future\_\_ import annotations

import pandas as pd

def normalized\_gain(pre: float, post: float) \-\> float:  
    if pre \>= 1.0:  
        return 0.0  
    return round((post \- pre) / (1 \- pre), 3\)

def engagement\_report(events: pd.DataFrame, quiz: pd.DataFrame) \-\> dict:  
    events \= events.sort\_values(\["user\_id", "ts"\]).copy()

\# Simple proxy for task time: only intervals up to 5 min  
    events\["next\_ts"\] \= events.groupby("user\_id")\["ts"\].shift(-1)  
    events\["delta\_s"\] \= (events\["next\_ts"\] \- events\["ts"\]).dt.total\_seconds()  
    events\["ontask\_s"\] \= events\["delta\_s"\].where(events\["delta\_s"\].between(1, 300), 0\)

    active\_events \= events\["event"\].isin(\["edit\_cell", "run\_cell", "submit\_exercise"\]).mean()

    retries \= (  
        events.query("event \== 'submit\_exercise'")  
        .groupby(\["user\_id", "exercise\_id"\])\["success"\]  
        .apply(lambda s: max(0, len(s) \- 1))  
        .mean()  
    )

    hint\_rate \= events\["event"\].eq("open\_hint").mean()

    quiz\_by\_user \= quiz.groupby("user\_id").agg(  
        pre=("pre\_score", "mean"),  
        post=("post\_score", "mean")  
    )  
    quiz\_by\_user\["gain"\] \= \[  
        normalized\_gain(pre, post)  
        for pre, post in zip(quiz\_by\_user\["pre"\], quiz\_by\_user\["post"\])  
    \]

    return {  
        "active\_coding\_ratio": round(float(active\_events), 3),  
        "mean\_retries\_per\_exercise": round(float(retries), 3),  
        "hint\_event\_ratio": round(float(hint\_rate), 3),  
        "median\_time\_on\_task\_min": round(events.groupby("user\_id")\["ontask\_s"\].sum().median() / 60, 2),  
        "mean\_normalized\_gain": round(float(quiz\_by\_user\["gain"\].mean()), 3),  
    }

events \= pd.DataFrame({  
    "user\_id": \[1,1,1,1,1,2,2,2\],  
    "ts": pd.to\_datetime(\[  
        "2026-05-01 10:00:00","2026-05-01 10:02:00","2026-05-01 10:03:00","2026-05-01 10:06:00",  
        "2026-05-01 10:08:00","2026-05-01 10:00:00","2026-05-01 10:04:00","2026-05-01 10:07:00"  
    \]),  
    "event": \["edit\_cell","run\_cell","open\_hint","submit\_exercise","submit\_exercise","edit\_cell","run\_cell","submit\_exercise"\],  
    "exercise\_id": \[None,None,None,"ex1","ex1",None,None,"ex1"\],  
    "success": \[None,None,None,False,True,None,None,True\]  
})

quiz \= pd.DataFrame({  
    "user\_id": \[1,2\],  
    "pre\_score": \[0.40, 0.55\],  
    "post\_score": \[0.75, 0.80\]  
})

print(engagement\_report(events, quiz))

Expected output:

{  
  'active\_coding\_ratio': 0.875,  
  'mean\_retries\_per\_exercise': 0.5,  
  'hint\_event\_ratio': 0.125,  
  'median\_time\_on\_task\_min': 5.0,  
  'mean\_normalized\_gain': 0.431  
}

For an agent, the golden rule is this:**Engagement metrics are useful for screening and improvement.**...not to assert causality of learning without the support of quizzes, transfer tests, or subsequent evaluation.[\[37\]](https://u.osu.edu/yuan.818/files/2020/09/Measuring-student-engagement-in-technology-mediated-learning-A-review-w-notes1.pdf)

## Agent pipeline and CI

If the tutorial will be delegated to an agent, the best arrangement is to separate roles. An agent**author**generates or revises content; an agent**evaluator**The system conducts technical and pedagogical checks; and the CI (Intelligent Communication) committee decides whether the material can be published. This division forces the system to differentiate between "looks good" and "passes the criteria." (GitHub documentation)[\[38\]](https://u.osu.edu/yuan.818/files/2020/09/Measuring-student-engagement-in-technology-mediated-learning-A-review-w-notes1.pdf)It shows native support for Python version arrays in GitHub Actions, which is sufficient to make this pipeline repeatable in PRs.[\[7\]](https://docs.github.com/actions/guides/building-and-testing-python)

### Prompt for the author agent

You are a technical tutorial author for a PyData-style Python library.

Aim:  
Create or revise a set of notebooks that teach the library of scientific computing to the general public.

Mandatory restrictions:  
\- Include H1, objectives, prerequisites, setup, dataset, exercises, and recap;  
\- Use solved examples with explicit sub-goals;  
\- Insert at least one recovery/prediction checkpoint every 10–15 minutes of equivalent content;  
\- Include immediate feedback via assertions, visible tests, or tips;  
\- Revisit at least two central concepts in a new context;  
\- Avoid redundancy between text and visuals;  
\- Support top-down execution in a clean kernel;  
\- Use a small dataset or "tiny" mode for CI;  
\- Write comments and markdown in Brazilian Portuguese;  
\- Tag exercises with metadata: {"role": "exercise"} when applicable.

Exit:  
\- notebooks reviewed;  
\- environment.yml;  
\- Local execution/Binder/Colab README;  
\- A short report explaining the pedagogical decisions.

### Prompt for the evaluating agent

You are a tutorial quality audit agent.

Task:  
Evaluate Python library tutorial notebooks.

Execute the following layers:  
1\. structural rubric;  
2\. linting de notebooks;  
3\. Clean execution on a new kernel;  
4\. Parameterized execution in tiny mode;  
5\. Unit testing and visual regression;  
6\. Time/memory budget;  
7\. Accessibility checks;  
8\. heuristic pedagogical score;  
9\. Final summary with PASS/FAIL.

Exit criteria:  
\- Produce JSON with scores per dimension;  
\- list blocking faults;  
\- List recommended improvements;  
\- Suggest specific corrections per notebook/cell;  
Never approve a notebook that fails in clean execution or reproducibility.

### Evaluation pipeline

flowchart LR  
A\[Tutorial Specification\] \--\> B\[Author Agent\]  
B \--\> C \[Structural checking and rubric scoring\]  
C \--\> D\[Notebook lint and style\]  
D \--\> E \[Clean execution with new kernel\]  
E \--\> F \[Parameterized tiny execution\]  
F \--\> G \[Unit testing and visual regression\]  
G \--\> H \[Time and memory budget\]  
H \--\> I \[Accessibility checks\]  
I \--\> J \[Pedagogical metrics and JSON report\]  
J \--\> K{Quality Gate}  
    K \-- FAIL \--\> B  
K \-- PASS \--\> L \[Publication in Jupyter Book\]  
    L \--\> M\[Links Binder e Colab\]

### Example of a workflow in GitHub Actions.

name: tutorial-quality

on:  
  pull\_request:  
  push:  
    branches: \[main\]

jobs:  
  validate:  
    runs-on: ubuntu-latest  
    strategy:  
      matrix:  
        python-version: \["3.10", "3.11", "3.12"\]

    steps:  
      \- uses: actions/checkout@v4

      \- uses: actions/setup-python@v5  
        with:  
          python-version: ${{ matrix.python-version }}

      \- name: Install  
        run: |  
          python \-m pip install \--upgrade pip  
          pip install \-e .\[tutorial\]  
          pip install nbqa ruff black nbval nbclient papermill pytest matplotlib jupytext

      \- name: Structural checks  
        run: |  
          python scripts/check\_structure.py tutorials/

      \- name: Notebook lint  
        run: |  
          nbqa ruff tutorials/\*.ipynb  
          nbqa black \--check tutorials/\*.ipynb

      \- name: Execute notebooks cleanly  
        run: |  
          python scripts/run\_clean.py tutorials/

      \- name: Parametrized tiny runs  
        run: |  
          python scripts/run\_tiny.py tutorials/

      \- name: Notebook validation  
        run: |  
          pytest \--nbval-lax tutorials/

      \- name: Unit and visual tests  
        run: |  
          pytest \-q tests/

      \- name: Performance budgets  
        run: |  
          python scripts/check\_budgets.py tutorials/

      \- name: Accessibility checks  
        run: |  
          python scripts/check\_accessibility.py tutorials/

      \- name: Score tutorial  
        run: |  
          python scripts/score\_tutorial.py \--out artifacts/tutorial\_report.json

      \- name: Upload report  
        uses: actions/upload-artifact@v4  
        with:  
          name: tutorial-report-${{ matrix.python-version }}  
          path: artifacts/

This template uses precisely the components described in the official documentation: Python array in GitHub Actions, notebook linting with nbQA, output collection and testing with nbval, and programmatic/parameterized execution with nbclient and papermill.[\[7\]](https://docs.github.com/actions/guides/building-and-testing-python)

### Suggested schedule

gantt  
Suggested timeline for producing and validating the tutorial.  
    dateFormat  YYYY-MM-DD  
    axisFormat  %d/%m

Scope section  
Define person, objectives, dataset :a1, 2026-05-11, 4d  
Sketch notebooks and checkpoints: a2, after a1, 3d

Production section  
Create solved example and exercises: b1, after a2, 5d  
Insert feedback, recap, and visual material: b2, after b1, 4d

Engineering section  
    Parear notebooks com Jupytext              :c1, after b1, 2d  
Configure environment, tests and CI: c2, after c1, 4d

Validation section  
Run clean execution and budgets: d1, then c2, 3d  
Review accessibility and UX: d2, after d1, 3d  
Driving with a small group: d3, after d2, 4d

section Publication  
    Publicar em Jupyter Book                   :e1, after d3, 2d  
    Make Binder/Colab available in release :e2, after e1, 2d

## UX, interactivity, and priority fonts

From a UX perspective, the most important rule is that the tutorial should be...**easy to navigate**, **easy to perform**and**easy to scan**For this, consistent headings are essential: the W3C reminds us that headings communicate structure and aid assistive technologies; the Berkeley Data Science Curriculum Guide recommends an H1 at the beginning of the notebook, alt text in markdown images, and alternative text paths for interpreting graphs; and the JupyterLab accessibility documentation notes that, for accessibility purposes, a more document-centric app is preferable to a multi-panel environment in some scenarios. In other words: don't rely solely on the graph; add a short textual summary, descriptive titles, correct headings, and clear labels.[\[39\]](https://www.w3.org/WAI/tutorials/page-structure/headings/)

In terms of visual design, apply multimedia theory conservatively. Prefer short blocks, one visual idea at a time, and simple diagrams of the library flow. A graphic should answer an explicit question; an animation or widget is only justified when the parameter variation changes understanding, and not just "makes it look cooler." The cutoff criterion is cognitive utility: if the interactive feature does not improve prediction, comparison, or interpretation, it is merely ornamentation.[\[40\]](https://www.jsu.edu/online/faculty/MULTIMEDIA%20LEARNING%20by%20Richard%20E.%20Mayer.pdf)

### Optional tools and trade-offs

The table below summarizes reasonable options for user experience and interactivity.

| Tool | Best use | Advantage | Risk/Limit | Recommendation |
| :---- | :---- | :---- | :---- | :---- |
| ipywidgets | Explore parameters, thresholds, hyperparameters, filters. | Immediate interaction on the laptop; good for "what if?" scenarios. | It can become a toy without a clear educational purpose. | Use only when there is a hypothesis to test. |
| Jupyter Book | Publish a collection of notebooks and markdown. | Navigation, TOC, citations, bibliography, reproducible build | Requires a publishing pipeline. | Ideal as a final format |
| Binder | Run tutorial in a replayable environment in the browser. | Almost no setup required for the participant. | Initial build and features vary; best with commit hash. | Excellent for public workshops. |
| Collaborate with Google[\[41\]](https://leomurta.github.io/papers/pimentel2019a.pdf) | No installation required, and quick to use in your browser. | Hosted service, very affordable for education. | Resource limitations and a less controlled environment | Good as a convenient mirror. |
| nbgrader | Assessable tasks and automatic feedback | Autograding structure and scoring by tests | Most useful when there is a formal collection of submissions. | Great for courses; optional for workshops. |

These recommendations are based on the official documentation for ipywidgets, Jupyter Book, Binder, Colab, and nbgrader.[\[42\]](https://ipywidgets.readthedocs.io/en/latest/)

### Annotated bibliography and prioritized sources.

The list below is intentionally short. It prioritizes what should truly guide an author or reviewer.

| Priority | Source | Why read first? |
| :---- | :---- | :---- |
| A | *Improving Students’ Learning With Effective Learning Techniques* (2013) [\[43\]](https://www.whz.de/fileadmin/lehre/hochschuldidaktik/docs/dunloskiimprovingstudentlearning.pdf) | Map of techniques of high and moderate utility; excellent for justifying retrieval, spacing, self-explanation, and interleaving practices. |
| A | *Distributed Practice in Verbal Recall Tasks* (2006) [\[44\]](https://augmentingcognition.com/assets/Cepeda2006.pdf) | Classic quantitative basis for spacing; useful for defending planned revisits throughout the session and the series of notebooks. |
| A | *The Effect of Testing Versus Restudy on Retention*(2014) and*The Critical Role of Retrieval Practice in Long-Term Retention* (2011) [\[45\]](https://courseware.epfl.ch/assets/courseware/v1/fdde2f0aa590bf3b1324077a6bf1540c/asset-v1%3AEPFL%2BDEMO%2B2020%2Btype%40asset%2Bblock/Rowland2014-meta-analysis.pdf) | A better set to justify short quizzes, predictions, recapitulation, and "memory" exercises instead of passive rereading. |
| A | *Retrieval Practice Produces More Learning than Elaborative Studying with Concept Mapping* (2011) [\[46\]](https://learninglab.psych.purdue.edu/downloads/2011/2011_Karpicke_Blunt_Science.pdf) | A powerful source to refute the illusion that any "active" activity is sufficient; it shows that the type of activity matters. |
| A | *The Use of Worked Examples as a Substitute for Problem Solving in Learning Algebra*(1985) and 2023 meta-analysis on worked examples[\[47\]](https://onderwijs.felienne.nl/vakdidactiek/materiaal/sweller_worked_examples.pdf) | They justify the use of solved examples, sub-objectives, and fading. |
| A | *The ICAP Framework*(2014) and meta-analysis of active learning in STEM (2014)[\[48\]](https://education.asu.edu/sites/g/files/litvpz656/files/lcl/chiwylie2014icap_2.pdf) | They support the design with code-along, frequent checks, and active participation instead of continuous exposure. |
| A | *Ten quick tips for creating an effective lesson* (2019), *Ten quick tips for teaching programming*(2018) and*Ten quick tips for delivering programming lessons* (2019) [\[49\]](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006915) | They transform research principles into operational criteria for pacing, short assessments, peer instruction, live coding, and pair programming. |
| A | *A Large-scale Study about Quality and Reproducibility of Jupyter Notebooks*(2019) and*Ten simple rules for writing and sharing computational analyses in Jupyter Notebooks* (2019) [\[50\]](https://leomurta.github.io/papers/pimentel2019a.pdf) | These are the most important sources for quality engineering in laptops. |
| B | *Teaching and Learning with Jupyter*and*The Turing Way* [\[51\]](https://users.ugent.be/~gvhaelew/doc/jupyter-edu-book.pdf) | They don't replace papers, but they greatly help in converting evidence into the architecture of teaching materials, publications, and collaboration. |
| B | *Role of Live-coding in Learning Introductory Programming* (2018) [\[52\]](https://pages.cs.wisc.edu/~gerald/papers/LiveCoding.pdf) | A good bridge between the general theory of engagement and the concrete practice of programming tutorials. |
| B | *The Power of Feedback*(2007) and 2020 feedback meta-analysis[\[53\]](https://conselhopedagogico.tecnico.ulisboa.pt/files/sites/32/hattie-and-timperley-2007.pdf) | Essential for calibrating automated feedback and avoiding vague or person-centric messages. |
| C | Literature in Portuguese on the testing effect, recall practice, and neuroeducation.[\[54\]](https://www.researchgate.net/publication/274359975_A_Systematic_Review_of_the_Testing_Effect_in_Learning) | Important for teacher training in Brazilian Portuguese and for communicating the scientific basis to local teams; less robust for effect estimates than the English corpus. |

In short, if the goal is to produce a really good PyData tutorial for your library, the best formula today is:**Interactive notebooks with serious software engineering, guided by retrieval practice, solved examples, immediate feedback and spaced revisits, and audited by an automated pipeline that treats pedagogy as something verifiable.**This combination best aligns evidence of learning, best practices from the notebook ecosystem, and the realistic expectations of the PyData community.[\[55\]](https://numfocus.org/programs/pydata)

---

[\[1\]](https://www.whz.de/fileadmin/lehre/hochschuldidaktik/docs/dunloskiimprovingstudentlearning.pdf) [\[12\]](https://www.whz.de/fileadmin/lehre/hochschuldidaktik/docs/dunloskiimprovingstudentlearning.pdf) [\[17\]](https://www.whz.de/fileadmin/lehre/hochschuldidaktik/docs/dunloskiimprovingstudentlearning.pdf) [\[43\]](https://www.whz.de/fileadmin/lehre/hochschuldidaktik/docs/dunloskiimprovingstudentlearning.pdf) https://www.whz.de/fileadmin/lehre/hochschuldidaktik/docs/dunloskiimprovingstudentlearning.pdf

[https://www.whz.de/fileadmin/lehre/hochschuldidaktik/docs/dunloskiimprovingstudentlearning.pdf](https://www.whz.de/fileadmin/lehre/hochschuldidaktik/docs/dunloskiimprovingstudentlearning.pdf)

[\[2\]](https://users.ugent.be/~gvhaelew/doc/jupyter-edu-book.pdf) [\[25\]](https://users.ugent.be/~gvhaelew/doc/jupyter-edu-book.pdf) [\[26\]](https://users.ugent.be/~gvhaelew/doc/jupyter-edu-book.pdf) [\[51\]](https://users.ugent.be/~gvhaelew/doc/jupyter-edu-book.pdf) https://users.ugent.be/\~gvhaelew/doc/jupyter-edu-book.pdf

[https://users.ugent.be/\~gvhaelew/doc/jupyter-edu-book.pdf](https://users.ugent.be/~gvhaelew/doc/jupyter-edu-book.pdf)

[\[3\]](https://onderwijs.felienne.nl/vakdidactiek/materiaal/sweller_worked_examples.pdf) [\[9\]](https://onderwijs.felienne.nl/vakdidactiek/materiaal/sweller_worked_examples.pdf) [\[47\]](https://onderwijs.felienne.nl/vakdidactiek/materiaal/sweller_worked_examples.pdf) https://onderwijs.felienne.nl/vakdidactiek/materiaal/sweller\_worked\_examples.pdf

[https://onderwijs.felienne.nl/vakdidactiek/materiaal/sweller\_worked\_examples.pdf](https://onderwijs.felienne.nl/vakdidactiek/materiaal/sweller_worked_examples.pdf)

[\[4\]](https://numfocus.org/programs/pydata) [\[55\]](https://numfocus.org/programs/pydata) https://numfocus.org/programs/pydata

[https://numfocus.org/programs/pydata](https://numfocus.org/programs/pydata)

[\[5\]](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006915) [\[28\]](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006915) [\[29\]](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006915) [\[32\]](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006915) [\[49\]](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006915) https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006915

[https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006915](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006915)

[\[6\]](https://u.osu.edu/yuan.818/files/2020/09/Measuring-student-engagement-in-technology-mediated-learning-A-review-w-notes1.pdf) [\[36\]](https://u.osu.edu/yuan.818/files/2020/09/Measuring-student-engagement-in-technology-mediated-learning-A-review-w-notes1.pdf) [\[37\]](https://u.osu.edu/yuan.818/files/2020/09/Measuring-student-engagement-in-technology-mediated-learning-A-review-w-notes1.pdf) [\[38\]](https://u.osu.edu/yuan.818/files/2020/09/Measuring-student-engagement-in-technology-mediated-learning-A-review-w-notes1.pdf) https://u.osu.edu/yuan.818/files/2020/09/Measuring-student-engagement-in-technology-mediated-learning-A-review-w-notes1.pdf

[https://u.osu.edu/yuan.818/files/2020/09/Measuring-student-engagement-in-technology-mediated-learning-A-review-w-notes1.pdf](https://u.osu.edu/yuan.818/files/2020/09/Measuring-student-engagement-in-technology-mediated-learning-A-review-w-notes1.pdf)

[\[7\]](https://docs.github.com/actions/guides/building-and-testing-python) https://docs.github.com/actions/guides/building-and-testing-python

[https://docs.github.com/actions/guides/building-and-testing-python](https://docs.github.com/actions/guides/building-and-testing-python)

[\[8\]](https://www.researchgate.net/publication/291826702_The_right_time_to_learn_Mechanisms_and_optimization_of_spaced_learning) [\[15\]](https://www.researchgate.net/publication/291826702_The_right_time_to_learn_Mechanisms_and_optimization_of_spaced_learning) https://www.researchgate.net/publication/291826702\_The\_right\_time\_to\_learn\_Mechanisms\_and\_optimization\_of\_spaced\_learning

[https://www.researchgate.net/publication/291826702\_The\_right\_time\_to\_learn\_Mechanisms\_and\_optimization\_of\_spaced\_learning](https://www.researchgate.net/publication/291826702_The_right_time_to_learn_Mechanisms_and_optimization_of_spaced_learning)

[\[10\]](https://journals.plos.org/ploscompbiol/article%3Fid%3D10.1371/journal.pcbi.1006023) https://journals.plos.org/ploscompbiol/article%3Fid%3D10.1371/journal.pcbi.1006023

[https://journals.plos.org/ploscompbiol/article%3Fid%3D10.1371/journal.pcbi.1006023](https://journals.plos.org/ploscompbiol/article%3Fid%3D10.1371/journal.pcbi.1006023)

[\[11\]](https://mrbartonmaths.com/resourcesnew/8.%20Research/Explicit%20Instruction/Structuring%20the%20Transition%20From%20Example%20Study%20to%20Problem%20Solving.pdf) https://mrbartonmaths.com/resourcesnew/8.%20Research/Explicit%20Instruction/Structuring%20the%20Transition%20From%20Example%20Study%20to%20Problem%20Solving.pdf

[https://mrbartonmaths.com/resourcesnew/8.%20Research/Explicit%20Instruction/Structuring%20the%20Transition%20From%20Example%20Study%20to%20Problem%20Solving.pdf](https://mrbartonmaths.com/resourcesnew/8.%20Research/Explicit%20Instruction/Structuring%20the%20Transition%20From%20Example%20Study%20to%20Problem%20Solving.pdf)

[\[13\]](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1007433) https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1007433

[https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1007433](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1007433)

[\[14\]](https://conselhopedagogico.tecnico.ulisboa.pt/files/sites/32/hattie-and-timperley-2007.pdf) [\[53\]](https://conselhopedagogico.tecnico.ulisboa.pt/files/sites/32/hattie-and-timperley-2007.pdf) https://conselhopedagogico.tecnico.ulisboa.pt/files/sites/32/hattie-and-timperley-2007.pdf

[https://conselhopedagogico.tecnico.ulisboa.pt/files/sites/32/hattie-and-timperley-2007.pdf](https://conselhopedagogico.tecnico.ulisboa.pt/files/sites/32/hattie-and-timperley-2007.pdf)

[\[16\]](https://augmentingcognition.com/assets/Cepeda2006.pdf) [\[44\]](https://augmentingcognition.com/assets/Cepeda2006.pdf) https://augmentingcognition.com/assets/Cepeda2006.pdf

[https://augmentingcognition.com/assets/Cepeda2006.pdf](https://augmentingcognition.com/assets/Cepeda2006.pdf)

[\[18\]](https://www.lscp.net/persons/ramus/docs/EPR20.pdf) https://www.lscp.net/persons/ramus/docs/EPR20.pdf

[https://www.lscp.net/persons/ramus/docs/EPR20.pdf](https://www.lscp.net/persons/ramus/docs/EPR20.pdf)

[\[19\]](https://education.asu.edu/sites/g/files/litvpz656/files/lcl/chiwylie2014icap_2.pdf) [\[21\]](https://education.asu.edu/sites/g/files/litvpz656/files/lcl/chiwylie2014icap_2.pdf) [\[48\]](https://education.asu.edu/sites/g/files/litvpz656/files/lcl/chiwylie2014icap_2.pdf) https://education.asu.edu/sites/g/files/litvpz656/files/lcl/chiwylie2014icap\_2.pdf

[https://education.asu.edu/sites/g/files/litvpz656/files/lcl/chiwylie2014icap\_2.pdf](https://education.asu.edu/sites/g/files/litvpz656/files/lcl/chiwylie2014icap_2.pdf)

[\[20\]](https://pages.cs.wisc.edu/~gerald/papers/LiveCoding.pdf) [\[52\]](https://pages.cs.wisc.edu/~gerald/papers/LiveCoding.pdf) https://pages.cs.wisc.edu/\~gerald/papers/LiveCoding.pdf

[https://pages.cs.wisc.edu/\~gerald/papers/LiveCoding.pdf](https://pages.cs.wisc.edu/~gerald/papers/LiveCoding.pdf)

[\[22\]](https://www.jsu.edu/online/faculty/MULTIMEDIA%20LEARNING%20by%20Richard%20E.%20Mayer.pdf) [\[23\]](https://www.jsu.edu/online/faculty/MULTIMEDIA%20LEARNING%20by%20Richard%20E.%20Mayer.pdf) [\[40\]](https://www.jsu.edu/online/faculty/MULTIMEDIA%20LEARNING%20by%20Richard%20E.%20Mayer.pdf) https://www.jsu.edu/online/faculty/MULTIMEDIA%20LEARNING%20by%20Richard%20E.%20Mayer.pdf

[https://www.jsu.edu/online/faculty/MULTIMEDIA%20LEARNING%20by%20Richard%20E.%20Mayer.pdf](https://www.jsu.edu/online/faculty/MULTIMEDIA%20LEARNING%20by%20Richard%20E.%20Mayer.pdf)

[\[24\]](https://www.researchgate.net/publication/274359975_A_Systematic_Review_of_the_Testing_Effect_in_Learning) [\[54\]](https://www.researchgate.net/publication/274359975_A_Systematic_Review_of_the_Testing_Effect_in_Learning) https://www.researchgate.net/publication/274359975\_A\_Systematic\_Review\_of\_the\_Testing\_Effect\_in\_Learning

[https://www.researchgate.net/publication/274359975\_A\_Systematic\_Review\_of\_the\_Testing\_Effect\_in\_Learning](https://www.researchgate.net/publication/274359975_A_Systematic_Review_of_the_Testing_Effect_in_Learning)

[\[27\]](https://mybinder.readthedocs.io/en/latest/tutorials/reproducibility.html) https://mybinder.readthedocs.io/en/latest/tutorials/reproducibility.html

[https://mybinder.readthedocs.io/en/latest/tutorials/reproducibility.html](https://mybinder.readthedocs.io/en/latest/tutorials/reproducibility.html)

[\[30\]](https://nbclient.readthedocs.io/en/latest/client.html) [\[33\]](https://nbclient.readthedocs.io/en/latest/client.html) https://nbclient.readthedocs.io/en/latest/client.html

[https://nbclient.readthedocs.io/en/latest/client.html](https://nbclient.readthedocs.io/en/latest/client.html)

[\[31\]](https://leomurta.github.io/papers/pimentel2019a.pdf) [\[35\]](https://leomurta.github.io/papers/pimentel2019a.pdf) [\[41\]](https://leomurta.github.io/papers/pimentel2019a.pdf) [\[50\]](https://leomurta.github.io/papers/pimentel2019a.pdf) https://leomurta.github.io/papers/pimentel2019a.pdf

[https://leomurta.github.io/papers/pimentel2019a.pdf](https://leomurta.github.io/papers/pimentel2019a.pdf)

[\[34\]](https://nbgrader.readthedocs.io/en/latest/user_guide/creating_and_grading_assignments.html) https://nbgrader.readthedocs.io/en/latest/user\_guide/creating\_and\_grading\_assignments.html

[https://nbgrader.readthedocs.io/en/latest/user\_guide/creating\_and\_grading\_assignments.html](https://nbgrader.readthedocs.io/en/latest/user_guide/creating_and_grading_assignments.html)

[\[39\]](https://www.w3.org/WAI/tutorials/page-structure/headings/) https://www.w3.org/WAI/tutorials/page-structure/headings/

[https://www.w3.org/WAI/tutorials/page-structure/headings/](https://www.w3.org/WAI/tutorials/page-structure/headings/)

[\[42\]](https://ipywidgets.readthedocs.io/en/latest/) https://ipywidgets.readthedocs.io/en/latest/

[https://ipywidgets.readthedocs.io/en/latest/](https://ipywidgets.readthedocs.io/en/latest/)

[\[45\]](https://courseware.epfl.ch/assets/courseware/v1/fdde2f0aa590bf3b1324077a6bf1540c/asset-v1%3AEPFL%2BDEMO%2B2020%2Btype%40asset%2Bblock/Rowland2014-meta-analysis.pdf) https://courseware.epfl.ch/assets/courseware/v1/fdde2f0aa590bf3b1324077a6bf1540c/asset-v1%3AEPFL%2BDEMO%2B2020%2Btype%40asset%2Bblock/Rowland2014-meta-analysis.pdf

[https://courseware.epfl.ch/assets/courseware/v1/fdde2f0aa590bf3b1324077a6bf1540c/asset-v1%3AEPFL%2BDEMO%2B2020%2Btype%40asset%2Bblock/Rowland2014-meta-analysis.pdf](https://courseware.epfl.ch/assets/courseware/v1/fdde2f0aa590bf3b1324077a6bf1540c/asset-v1%3AEPFL%2BDEMO%2B2020%2Btype%40asset%2Bblock/Rowland2014-meta-analysis.pdf)

[\[46\]](https://learninglab.psych.purdue.edu/downloads/2011/2011_Karpicke_Blunt_Science.pdf) https://learninglab.psych.purdue.edu/downloads/2011/2011\_Karpicke\_Blunt\_Science.pdf

[https://learninglab.psych.purdue.edu/downloads/2011/2011\_Karpicke\_Blunt\_Science.pdf](https://learninglab.psych.purdue.edu/downloads/2011/2011_Karpicke_Blunt_Science.pdf)