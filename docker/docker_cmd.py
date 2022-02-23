#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import logging
import os
import subprocess

logging.basicConfig(level=logging.DEBUG)

DIR = os.path.dirname(os.path.realpath(__file__))


def build(
    image="slaid",
    lib_version="",
    docker_build_dir="../docker-build",
    docker_args=None,
    build_args=None,
    extra_tags=None,
):
    if not os.path.exists(docker_build_dir):
        os.mkdir(docker_build_dir)

    build_args = build_args or []
    kwargs_list = [
        dict(
            dockerfile="Dockerfile",
            build_dir=docker_build_dir,
            image=image,
            docker_args=docker_args,
            build_args=build_args,
        ),
        dict(
            dockerfile="Dockerfile",
            build_dir=docker_build_dir,
            image=image,
            tag=f"{lib_version}",
            docker_args=docker_args,
            build_args=build_args,
        ),
    ]

    for model_path, model_name in get_models():
        tag = _get_tag(lib_version, model_name, extra_tags)
        kwargs_list.append(
            dict(
                build_dir=docker_build_dir,
                image=image,
                tag=tag,
                build_args=[f"MODEL={model_path}"] + build_args,
                docker_args=docker_args,
            ))

    for kwargs in kwargs_list:
        docker_build(**kwargs)


def docker_build(
    build_dir,
    image,
    tag="",
    build_args=None,
    docker_args="",
    dockerfile="Dockerfile",
):
    build_args = build_args or []
    build_args = (f'{" ".join(["--build-arg " + arg for arg in build_args])}'
                  if build_args else "")
    command = (f"docker {docker_args}  build -f {dockerfile} "
               f'-t {image}{":" + tag if tag else ""} '
               f"{build_dir} "
               f"{build_args}")

    logging.debug(command)
    subprocess.run(command, shell=True, check=True)


def push(image, **kwargs):
    command = f"docker  push {image}"
    logging.debug(command)
    #  docker_push(**kwargs)


def docker_push(image, tag, repo, docker_args=""):
    command = (f'docker {docker_args} push {repo + "/" if repo else ""}{image}'
               f'{":" + tag if tag else ""}')
    logging.debug(command)
    subprocess.run(command, shell=True, check=True)


def tag(repo, image="slaid", lib_version="", docker_args="", extra_tags=None):
    kwargs_list = [
        dict(repo=repo,
             image=image,
             tag=f"{lib_version}",
             docker_args=docker_args)
    ]

    for _, model_name in get_models():
        tag = _get_tag(lib_version, model_name, extra_tags)
        kwargs_list.append(
            dict(
                repo=repo,
                image=image,
                tag=tag,
                docker_args=docker_args,
            ))

    for kwargs in kwargs_list:
        print(docker_tag(**kwargs))


def docker_tag(repo, image, tag, docker_args=""):
    tag = ":" + tag if tag else ""
    final_tag = f"{repo}/{image}{tag}"
    command = f"docker {docker_args} tag {image}{tag} {final_tag}"
    logging.debug(command)
    subprocess.run(command, shell=True, check=True)
    return final_tag


def _get_tag(lib_version, model_name, extra_tags=None):
    extra_tags = extra_tags or []
    tag = f'{lib_version + "-" if lib_version else ""}{model_name}'
    for extra_tag in extra_tags:
        tag += f"-{extra_tag}"
    return tag


def get_models():
    with open("filter-models.txt", "r") as f_obj:
        models = f_obj.read().splitlines()
    for model_path in models:
        model_path = retrieve_model(model_path)
        model_no_ext = os.path.splitext(model_path)[0]
        yield os.path.basename(model_path), model_no_ext


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", dest="lib_version", default="")
    parser.add_argument("-a",
                        dest="docker_args",
                        help="docker args like host, ect.",
                        default="")
    subparsers = parser.add_subparsers()

    build_parser = subparsers.add_parser("build")
    build_parser.set_defaults(func=build)
    build_parser.add_argument("-d",
                              dest="docker_build_dir",
                              default="../docker-build")

    comma_separated_list_handler = lambda s: [i for i in s.split(",")]
    build_parser.add_argument("-e",
                              dest="extra_tags",
                              type=comma_separated_list_handler)
    build_parser.add_argument("--build-args",
                              dest="build_args",
                              type=comma_separated_list_handler)

    push_parser = subparsers.add_parser("push")
    push_parser.add_argument("image", action="append")
    push_parser.set_defaults(func=push)

    tag_parser = subparsers.add_parser("tag")
    tag_parser.add_argument("-r", dest="repo")
    tag_parser.add_argument("-e",
                            dest="extra_tags",
                            type=comma_separated_list_handler)
    tag_parser.set_defaults(func=tag)

    args = parser.parse_args()
    args_dct = vars(args)
    func = args_dct.pop("func")
    func(**args_dct)
