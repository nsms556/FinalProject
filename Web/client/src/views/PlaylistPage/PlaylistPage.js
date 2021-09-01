import React from "react";
import {Card, CardHeader, Container, Row} from "reactstrap";

import Header from "../../components/Headers/Header";

import SongInfo from "./Sections/SongInfo";


function PlaylistPage() {
    return (
        <>
            <Header/>
            <Container className="mt--7" fluid>
                <Row>
                    <div className="col">
                        <Card className="shadow">
                            <CardHeader className="border-0">
                                <h3 className="mb-0">My Playlist</h3>
                            </CardHeader>
                            {/* Song Info */}
                            <SongInfo/>
                        </Card>
                    </div>
                </Row>
            </Container>
        </>
    );
};

export default PlaylistPage;
