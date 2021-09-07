import React from "react";
import {
    Button,
    Card,
    CardHeader,
    Col,
    Container,
    Input,
    InputGroup,
    InputGroupAddon,
    InputGroupText,
    Row
} from "reactstrap";

import Header from "../../components/Headers/Header";

import SongInfo from "../../components/SongItems/SongInfo";


function SearchPage() {
    return (
        <>
            <Header/>
            <Container className="mt--7" fluid>
                <Row>
                    <div className="col">
                        <Card className="shadow">
                            <CardHeader className="border-0">
                                <Row>
                                    <Col xs="9">
                                        <InputGroup className="mb-4">
                                            <InputGroupAddon addonType="prepend">
                                                <InputGroupText>
                                                    <i className="ni ni-note-03"/>
                                                </InputGroupText>
                                            </InputGroupAddon>
                                            <Input placeholder="Search" type="text"/>
                                        </InputGroup>
                                    </Col>
                                    <Col xs="1">
                                        <Button color="info" outline type="button" className="mb-4">
                                            Search
                                        </Button>
                                    </Col>
                                </Row>
                            </CardHeader>
                            {/* Song Info */}
                            <SongInfo/>
                        </Card>
                    </div>
                </Row>
            </Container>
        </>
    );
}

export default SearchPage;
