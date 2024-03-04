# AvatAIr<br />
<img src="https://raw.githubusercontent.com/lukassteinwender/avatair/main/Documentation/picture/screenshot.png" width="1000"><br />
<br />

**Installation:**<br />
```
pip install -r requirements.txt
```

**Script:**<br />
```
python avatair.py
```

**Config:**<br />
<table>
  <thead>
    <tr>
      <th><b>Name</b></th>
      <th><b>Description</b></th>
      <th><b>Default</b></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th><b>initial</b></th>
      <th>Number of iterations that will run until the program is finished</th>
      <th>5</th>
    </tr>
    <tr>
      <th><b>scales</b></th>
      <th>3 types of scales for other use-cases
      <br />1 = acceptance, likeability, empathy, anthropomorphism, trust
      <br />2 = openness, conscientiousness, extraversion, agreeableness, neuroticism
      <br />3 = efficiency</th>
      <th>1</th>
    </tr>
    <tr>
      <th><b>pictures</b></th>
      <th>The amount of pictures should be generated during one iteration (Int)</th>
      <th>1</th>
    </tr>
    <tr>
      <th><b>attention</b></th>
      <th>Possibility to add several attention-checks after x iterations
      (Array (e.g [1,2,3]), [-1] for no check)<br /></th>
      <th>[1,3]</th>
    </tr>
    <tr>
      <th><b>stablediffusion</b></th>
      <th>Choose between Stable Diffusion ("sd") and Stable Diffusion XL ("xl")</th>
      <th>"xl"</th>
    </tr>
    <tr>
      <th><b>model</b></th>
      <th>Use different types of SD-models</th>
      <th>"SG161222/RealVisXL_V3.0"</th>
    </tr>
    <tr>
      <th><b>token</b></th>
      <th>HuggingFace login-token, just needed for some specific SD-models</th>
      <th>""</th>
    </tr>
    <tr>
      <th><b>promptmodel</b></th>
      <th>Generate prompt with defined or latent variables 
        <br />("defined" or "latent")
        <br />defined = e.g. abstraction, haircolor, eyecolor, ...
        <br />latent= e.g. acceptance, likeability, ...</th>
      <th>defined</th>
    </tr>
  </tbody>
</table>
<br />


**Requirements:**<br />
requirements.txt

**The pipeline works as following:**<br /><br />
<img src="https://raw.githubusercontent.com/lukassteinwender/avatair/main/Documentation/picture/pipeline.png" width="600"><br />

<br />

**Paper:**<br />
Download the paper <a href="https://steinwender-media.com/blog/2023/10/24/avatair/">here</a>.

**License:**<br />
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

**For non-commercial use only!**
