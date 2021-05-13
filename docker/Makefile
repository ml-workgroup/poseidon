NAME="hbwinther/poseidon"
DOCKER=docker
ABSPWD=$(shell cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )
PROJECTDIR=$(shell dirname "$(ABSPWD)")

.PHONY: latest tagged list push clean prune


latest:
	$(DOCKER) build -t $(NAME):latest .


.ONESHELL:
tagged:
	@echo "example: 1.7.1_1  //poseidon-version";
	@read -p "enter TAG: $(NAME):" TAG;
	$(DOCKER) build -t $(NAME):$$TAG .


list:
	docker images $(NAME)


.ONESHELL:
push: list
	@read -p "enter tag to push: " TAG;
	$(DOCKER) push $(NAME):$$TAG


clean:
	$(DOCKER) rmi --force $(NAME)


prune:
	$(DOCKER) system prune -a


run:
	docker run -it --rm \
		-p 5000:80 \
		-v "$(PROJECTDIR)/src:/poseidon" \
		-v "$(ABSPWD)/uwsgi.ini:/app/uwsgi.ini" \
		-v "$(PROJECTDIR)/example_models:/models" \
		$(NAME)


dev:
	docker run -it --rm \
		-p 5000:5000 \
		-v "$(PROJECTDIR)/src:/poseidon" \
		-v "$(ABSPWD)/uwsgi.ini:/app/uwsgi.ini" \
		-v "$(PROJECTDIR)/example_models:/models" \
		$(NAME) bash -c "\
			cd /poseidon && \
			python main.py \
		"


interactive:
	docker run -it --rm \
		-p 8080:80 \
		-v "$(PROJECTDIR)/src:/poseidon" \
		-v "$(ABSPWD)/uwsgi.ini:/app/uwsgi.ini" \
		-v "$(PROJECTDIR)/example_models:/models" \
		$(NAME) bash
