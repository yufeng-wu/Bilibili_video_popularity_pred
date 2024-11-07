# Deep Learning for Video Popularity Prediction

Author: Yufeng Wu

Course: Williams College CSCI 374: Machine Learning

Date: December 2022


### Repo Organization

- `data/`: contains raw and processed bilibili video meta data.
- `BiliSpider.py`: defines a class BiliSpider, which is a worker that scrapes video stats from Bilibili.
- `tag_names.txt`: a list of random tags (in Chinese) used to search for videos on Bilibili.
- `scraper.py`: scrapes data from Bilibili, using the BiliSpider class.
- `convert_raw_txt_to_csv.py`: helper program to convert raw output from `scraper.py` into csv formats.
- `analysis.py`: train, tune, and test machine learning models to predict the popularity of 
Bilibili videos.
- `paper.pdf`: technical report of this project.
